// SPDX-License-Identifier: GPL-3.0-or-later
//! Best-effort hardware identification. Phase 4 derives the GPU name from the
//! backend's device banner (llama.cpp prints it); vendor is inferred from the name.
//! VRAM/bandwidth are left to a later pass (per-OS tools) and default to 0.

/// Infer the vendor bucket from a device name.
pub fn vendor_of(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("nvidia") || n.contains("geforce") || n.contains("rtx") || n.contains("tesla") {
        "NVIDIA"
    } else if n.contains("amd") || n.contains("radeon") || n.contains("instinct") {
        "AMD"
    } else if n.contains("apple")
        || n.contains(" m1")
        || n.contains(" m2")
        || n.contains(" m3")
        || n.contains(" m4")
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
        assert_eq!(vendor_of("Apple M4 Max"), "Apple");
        assert_eq!(vendor_of("Intel(R) UHD Graphics 630"), "Intel");
        assert_eq!(vendor_of("Ryzen 9 7950X"), "CPU");
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
