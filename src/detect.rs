// SPDX-License-Identifier: GPL-3.0-or-later
//! Best-effort hardware identification. Phase 4 derives the GPU name from the
//! backend's device banner (llama.cpp prints it); vendor is inferred from the name.
//! VRAM/bandwidth are left to a later pass (per-OS tools) and default to 0.

/// Infer the vendor bucket from a device name.
pub fn vendor_of(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("nvidia")
        || n.contains("geforce")
        || n.contains("rtx")
        || n.contains("tesla")
        || n.contains("cmp ")
    {
        "NVIDIA"
    } else if n.contains("amd") || n.contains("radeon") || n.contains("instinct") {
        "AMD"
    } else if n.contains("apple")
        || n.contains(" m1")
        || n.contains(" m2")
        || n.contains(" m3")
        || n.contains(" m4")
        || n.contains(" m5")
    {
        "Apple"
    } else if n.contains("intel") {
        "Intel"
    } else {
        "CPU"
    }
}

/// Best-effort GPU name from `nvidia-smi`, used as a fallback when the backend init banner
/// didn't yield a device name (e.g. a build whose device line we don't recognize). Returns
/// the first GPU's name, or None if nvidia-smi is absent/fails. Only call this when the run
/// actually used the GPU, so a CPU-only run isn't mislabeled as the installed card.
pub fn nvidia_gpu_name() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
}

#[derive(Debug, PartialEq, Eq)]
struct NvidiaGpu {
    uuid: String,
    name: String,
    memory_mib: u64,
}

fn parse_nvidia_smi(output: &str) -> Vec<NvidiaGpu> {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(3, ',').map(str::trim);
            Some(NvidiaGpu {
                uuid: fields.next()?.to_string(),
                name: fields.next()?.to_string(),
                memory_mib: fields.next()?.parse().ok()?,
            })
        })
        .collect()
}

fn nvidia_smi_group(
    output: &str,
    device_name: &str,
    cuda_visible_devices: Option<&str>,
) -> Option<(usize, u64)> {
    let gpus = parse_nvidia_smi(output);
    let visible: Vec<&NvidiaGpu> = match cuda_visible_devices.map(str::trim) {
        None | Some("") | Some("all") => gpus.iter().collect(),
        Some("-1") => Vec::new(),
        Some(value) => {
            let mut selected = Vec::new();
            for identity in value.split(',').map(str::trim) {
                if identity.to_ascii_uppercase().starts_with("MIG-") {
                    return None;
                }
                let gpu = identity
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| gpus.get(index))
                    .or_else(|| unique_by_identity(&gpus, identity, |gpu| &gpu.uuid))?;
                if !selected.contains(&gpu) {
                    selected.push(gpu);
                }
            }
            selected
        }
    };
    let matches: Vec<_> = visible
        .into_iter()
        .filter(|gpu| gpu.name.eq_ignore_ascii_case(device_name.trim()))
        .collect();
    let count = matches.len();
    let total_bytes = matches.iter().try_fold(0_u64, |total, gpu| {
        total.checked_add(gpu.memory_mib.checked_mul(1024 * 1024)?)
    })?;
    Some((count, bytes_to_rounded_gib(total_bytes)?))
}

#[derive(Debug, PartialEq, Eq)]
struct NvidiaMig {
    uuid: String,
    memory_gb: u64,
}

fn parse_nvidia_mig_list(output: &str) -> Vec<NvidiaMig> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let details = line.strip_prefix("MIG ")?;
            let profile = details.split_whitespace().next()?;
            let memory_gb = profile.split('.').find_map(|part| {
                let lower = part.to_ascii_lowercase();
                let (value, _) = lower.split_once("gb")?;
                value.parse::<u64>().ok()
            })?;
            let (_, uuid) = line.rsplit_once("(UUID: ")?;
            let uuid = uuid.strip_suffix(')')?.trim();
            Some(NvidiaMig {
                uuid: uuid.to_string(),
                memory_gb,
            })
        })
        .collect()
}

fn selector_index(selector: &str) -> Option<usize> {
    let digits = selector
        .trim()
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn uuid_matches(uuid: &str, identity: &str) -> bool {
    uuid.eq_ignore_ascii_case(identity) || uuid.to_lowercase().starts_with(&identity.to_lowercase())
}

fn unique_by_identity<'a, T>(
    values: &'a [T],
    identity: &str,
    uuid: impl Fn(&T) -> &str,
) -> Option<&'a T> {
    let mut matches = values
        .iter()
        .filter(|value| uuid_matches(uuid(value), identity));
    let only = matches.next()?;
    matches.next().is_none().then_some(only)
}

fn nvidia_smi_vram_gb(
    output: &str,
    mig_output: &str,
    device_name: &str,
    selected_device: Option<&str>,
    cuda_visible_devices: Option<&str>,
    cuda_backend: bool,
) -> Option<u64> {
    let gpus = parse_nvidia_smi(output);
    let mig_devices = parse_nvidia_mig_list(mig_output);
    let selected = selected_device
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            if cuda_backend {
                cuda_visible_devices
                    .filter(|visible| !visible.trim().is_empty())
                    .map(|_| "CUDA0")
            } else {
                None
            }
        });

    let selected_identity = selected.and_then(|selector| {
        let upper = selector.to_ascii_uppercase();
        if upper.starts_with("GPU-") || upper.starts_with("MIG-") {
            return Some(selector);
        }
        if !upper.starts_with("CUDA") {
            return None;
        }
        let logical_index = selector_index(selector)?;
        cuda_visible_devices
            .and_then(|visible| visible.split(',').nth(logical_index))
            .map(str::trim)
            .filter(|identity| {
                let upper = identity.to_ascii_uppercase();
                upper.starts_with("GPU-") || upper.starts_with("MIG-")
            })
    });

    if let Some(identity) = selected_identity {
        if identity.to_ascii_uppercase().starts_with("MIG-") {
            return unique_by_identity(&mig_devices, identity, |mig| &mig.uuid)
                .map(|mig| mig.memory_gb);
        }
        return unique_by_identity(&gpus, identity, |gpu| &gpu.uuid)
            .filter(|gpu| gpu.name.eq_ignore_ascii_case(device_name.trim()))
            .and_then(|gpu| bytes_to_rounded_gib(gpu.memory_mib.saturating_mul(1024 * 1024)));
    }

    let mut matches = gpus
        .iter()
        .filter(|gpu| gpu.name.eq_ignore_ascii_case(device_name.trim()));
    let gpu = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    bytes_to_rounded_gib(gpu.memory_mib.saturating_mul(1024 * 1024))
}

/// Installed VRAM for the selected NVIDIA device, rounded to whole GiB. This
/// disambiguates cards such as the RTX 4060 Ti whose 8 GB and 16 GB variants
/// report the same model name.
pub fn nvidia_vram_gb(
    device_name: &str,
    selected_device: Option<&str>,
    backend_label: &str,
) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let cuda_backend = backend_label
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .any(|part| part.eq_ignore_ascii_case("CUDA"));
    let mig_list = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    nvidia_smi_vram_gb(
        &String::from_utf8_lossy(&output.stdout),
        &mig_list,
        device_name,
        selected_device,
        visible.as_deref(),
        cuda_backend,
    )
}

/// Count equally named visible CUDA devices and sum their installed VRAM. With no
/// explicit device selector, llama.cpp splits an offloaded model across all visible
/// CUDA devices; recording the group prevents a multi-GPU result from masquerading
/// as a single-card result.
pub fn nvidia_gpu_group(
    device_name: &str,
    selected_device: Option<&str>,
    backend_label: &str,
) -> Option<(usize, u64)> {
    if selected_device.is_some()
        || !backend_label
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .any(|part| part.eq_ignore_ascii_case("CUDA"))
    {
        return None;
    }
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    output.status.success().then(|| {
        nvidia_smi_group(
            &String::from_utf8_lossy(&output.stdout),
            device_name,
            visible.as_deref(),
        )
    })?
}

#[cfg(target_os = "macos")]
fn sysctl(key: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// The Apple-silicon chip name via sysctl (e.g. "Apple M4"), or None off macOS / non-Apple.
/// This is authoritative for Apple GPUs and clean, unlike the Metal banner which can read
/// as "MTL0 (Apple M4)".
pub fn apple_chip() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl("machdep.cpu.brand_string").filter(|n| n.starts_with("Apple")) {
            return Some(n);
        }
    }
    None
}

/// Best-effort accelerator name when the backend banner didn't yield one: nvidia-smi for
/// NVIDIA, then the Apple-silicon chip via sysctl. None on a plain CPU box.
pub fn gpu_name() -> Option<String> {
    nvidia_gpu_name().or_else(apple_chip)
}

/// The host CPU model as the OS reports it, e.g. "AMD EPYC 7J13 64-Core Processor".
/// Best-effort: /proc/cpuinfo on Linux, sysctl on macOS, PROCESSOR_IDENTIFIER on
/// Windows. None when nothing usable is found — the submission field is optional.
pub fn cpu_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines()
            .find_map(|line| line.strip_prefix("model name"))
            .map(|v| clean_spaces(v.trim_start_matches([' ', '\t', ':'])))
            .filter(|s| !s.is_empty())
    }
    #[cfg(target_os = "macos")]
    {
        sysctl("machdep.cpu.brand_string").map(|s| clean_spaces(&s))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("PROCESSOR_IDENTIFIER")
            .ok()
            .map(|s| clean_spaces(&s))
            .filter(|s| !s.is_empty())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

fn bytes_to_rounded_gib(bytes: u64) -> Option<u64> {
    (bytes > 0).then_some((bytes.saturating_add(BYTES_PER_GIB / 2) / BYTES_PER_GIB).max(1))
}

#[cfg(target_os = "linux")]
fn linux_meminfo_gib(info: &str) -> Option<u64> {
    let kib = info.lines().find_map(|line| {
        let value = line.strip_prefix("MemTotal:")?.trim();
        let mut parts = value.split_whitespace();
        let kib = parts.next()?.parse::<u64>().ok()?;
        parts.next()?.eq_ignore_ascii_case("kb").then_some(kib)
    })?;
    bytes_to_rounded_gib(kib.saturating_mul(1024))
}

/// Total physical system memory in GiB, rounded to the nearest whole GiB.
/// Best-effort: /proc/meminfo on Linux, hw.memsize on macOS, and CIM through
/// PowerShell on Windows. None keeps the additive contract field absent.
pub fn system_ram_gb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        linux_meminfo_gib(&info)
    }
    #[cfg(target_os = "macos")]
    {
        let bytes = sysctl("hw.memsize")?.parse::<u64>().ok()?;
        bytes_to_rounded_gib(bytes)
    }
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let bytes = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
            .ok()?;
        bytes_to_rounded_gib(bytes)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Collapse runs of whitespace — Intel brand strings pad with doubled spaces.
fn clean_spaces(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Apple unified memory (≈ usable GPU memory) in GB, via sysctl. 0 off macOS / on failure.
pub fn apple_unified_mem_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = sysctl("hw.memsize").and_then(|s| s.parse::<f64>().ok()) {
            return (bytes / 1e9).round();
        }
    }
    0.0
}

/// A stable, lowercase, dash-separated slug for an id.
pub fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_inference() {
        assert_eq!(vendor_of("AMD Radeon Pro 5500M (MoltenVK)"), "AMD");
        assert_eq!(vendor_of("NVIDIA GeForce RTX 4090"), "NVIDIA");
        assert_eq!(vendor_of("CMP 170HX"), "NVIDIA");
        assert_eq!(vendor_of("Apple M4 Max"), "Apple");
        assert_eq!(vendor_of("Apple M5 Pro"), "Apple");
        assert_eq!(vendor_of("Intel(R) UHD Graphics 630"), "Intel");
        assert_eq!(vendor_of("Ryzen 9 7950X"), "CPU");
    }

    #[test]
    fn groups_same_model_nvidia_devices_and_sums_vram() {
        let output = "GPU-aaaa, CMP 170HX, 65536\n\
                      GPU-bbbb, CMP 170HX, 65536\n\
                      GPU-cccc, NVIDIA H100, 81920\n";
        assert_eq!(nvidia_smi_group(output, "CMP 170HX", None), Some((2, 128)));
        assert_eq!(
            nvidia_smi_group(output, "CMP 170HX", Some("GPU-bbbb")),
            Some((1, 64))
        );
        assert_eq!(
            nvidia_smi_group(output, "CMP 170HX", Some("1,2")),
            Some((1, 64))
        );
        assert_eq!(nvidia_smi_group(output, "NVIDIA H100", None), Some((1, 80)));
        assert_eq!(nvidia_smi_group(output, "NVIDIA A100", None), None);
        assert_eq!(nvidia_smi_group(output, "CMP 170HX", Some("-1")), None);
    }

    #[test]
    fn cpu_brand_strings_collapse_padding() {
        assert_eq!(
            clean_spaces("Intel(R) Xeon(R) CPU  E5-2680 v4  @  2.40GHz"),
            "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"
        );
    }

    #[test]
    fn rounds_physical_memory_to_whole_gibibytes() {
        assert_eq!(bytes_to_rounded_gib(16 * BYTES_PER_GIB), Some(16));
        assert_eq!(bytes_to_rounded_gib(0), None);
    }

    #[test]
    fn parses_nvidia_memory_for_the_selected_device() {
        let output = "GPU-aaaaaaaa, NVIDIA GeForce RTX 4060 Ti, 8188\n\
                      GPU-bbbbbbbb, NVIDIA GeForce RTX 4060 Ti, 16380\n\
                      GPU-cccccccc, NVIDIA GeForce RTX 4090, 24564\n";
        let mig_output = "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-dddddddd)\n\
                          MIG 1c.3g.40gb Device 0: (UUID: MIG-eeeeeeee)\n";
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                Some("CUDA1"),
                None,
                true,
            ),
            None
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                Some("CUDA0"),
                Some("GPU-bbbbbbbb,GPU-aaaaaaaa,GPU-cccccccc"),
                true,
            ),
            Some(16)
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                None,
                Some("GPU-bbbbbbbb"),
                true,
            ),
            Some(16)
        );
        // CUDA visibility may be inherited by a non-CUDA run; reject name mismatches.
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                None,
                Some("GPU-cccccccc"),
                false,
            ),
            None
        );
        // Numeric CUDA ordinals are not nvidia-smi indices; never cross-map them.
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                None,
                Some("1"),
                true,
            ),
            None
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                Some("GPU-aaaaaaaa"),
                None,
                false,
            ),
            Some(8)
        );
        // A stable UUID must never fall back to a different uniquely named GPU.
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4090",
                Some("GPU-aaaaaaaa"),
                None,
                false,
            ),
            None
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4090",
                None,
                None,
                false,
            ),
            Some(24)
        );
        // A non-CUDA backend's ordinal is unrelated to nvidia-smi's index.
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4090",
                Some("Vulkan1"),
                None,
                false,
            ),
            Some(24)
        );
        // Duplicate names are ambiguous without a selector; never guess the variant.
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                None,
                None,
                false,
            ),
            None
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA GeForce RTX 4060 Ti",
                Some("Vulkan1"),
                None,
                false,
            ),
            None
        );
        assert_eq!(
            nvidia_smi_vram_gb(
                output,
                mig_output,
                "NVIDIA A100-SXM4-80GB MIG 3g.40gb",
                None,
                Some("MIG-eeeeeeee"),
                true,
            ),
            Some(40)
        );
        // A missing MIG UUID must not fall back to the parent GPU's full memory.
        assert_eq!(
            nvidia_smi_vram_gb(
                "GPU-dddddddd, NVIDIA A100-SXM4-80GB, 81920\n",
                mig_output,
                "NVIDIA A100-SXM4-80GB",
                None,
                Some("MIG-missing"),
                true,
            ),
            None
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn parses_linux_memtotal() {
        let info = "MemTotal:       131072000 kB\nMemFree:          123456 kB\n";
        assert_eq!(linux_meminfo_gib(info), Some(125));
        assert_eq!(linux_meminfo_gib("MemFree: 10 kB\n"), None);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn cpu_name_detects_something_on_dev_and_ci_hosts() {
        let name = cpu_name().expect("linux/macos should always yield a CPU model");
        assert!(!name.trim().is_empty());
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn system_ram_detects_something_on_dev_and_ci_hosts() {
        assert!(system_ram_gb().is_some_and(|gib| gib > 0));
    }

    #[test]
    fn slugs() {
        assert_eq!(
            slugify("AMD Radeon Pro 5500M (MoltenVK)"),
            "amd-radeon-pro-5500m-moltenvk"
        );
        assert_eq!(slugify("Qwen3.5 4B"), "qwen3-5-4b");
    }
}
