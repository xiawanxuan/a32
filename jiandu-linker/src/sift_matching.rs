use crate::models::{BambooSlip, SiftFeature, EdgeFeatures};
use std::collections::HashMap;

const DESCRIPTOR_DIM: usize = 128;
const KEYPOINTS_PER_EDGE: usize = 8;

pub struct SiftMatcher {
    sift_cache: HashMap<String, EdgeFeatures>,
    random_seed: u64,
}

impl SiftMatcher {
    pub fn new() -> Self {
        SiftMatcher {
            sift_cache: HashMap::new(),
            random_seed: 42,
        }
    }

    fn pseudo_random(&self, seed: u64, step: u64) -> f64 {
        let mut x = seed.wrapping_add(step).wrapping_mul(6364136223846793005);
        x = x.wrapping_add(1442695040888963407);
        x = (x ^ (x >> 22)) ^ ((x ^ (x >> 22)) >> 17) ^ x;
        (x as u64 % 1000000) as f64 / 1000000.0
    }

    fn hash_string(&self, s: &str) -> u64 {
        let mut hash: u64 = self.random_seed;
        for c in s.chars() {
            hash = hash.wrapping_add(c as u64);
            hash = hash.wrapping_mul(31);
        }
        hash
    }

    pub fn extract_sift_features(&mut self, slip: &BambooSlip) -> EdgeFeatures {
        if let Some(cached) = self.sift_cache.get(&slip.slip_id) {
            return cached.clone();
        }

        let slip_hash = self.hash_string(&slip.slip_id);
        let glyph_hash = self.hash_string(&slip.glyph_features);
        let stroke_hash = self.hash_string(&slip.remnant_strokes);

        let left_edge = self.generate_edge_keypoints(
            slip_hash,
            glyph_hash,
            stroke_hash,
            &slip.glyph_features,
            true,
        );

        let right_edge = self.generate_edge_keypoints(
            slip_hash.wrapping_add(1000),
            glyph_hash.wrapping_add(1000),
            stroke_hash.wrapping_add(1000),
            &slip.glyph_features,
            false,
        );

        let stroke_continuation = self.extract_stroke_continuation(slip);

        let features = EdgeFeatures {
            slip_id: slip.slip_id.clone(),
            left_edge,
            right_edge,
            stroke_continuation,
        };

        self.sift_cache.insert(slip.slip_id.clone(), features.clone());
        features
    }

    fn generate_edge_keypoints(
        &self,
        slip_seed: u64,
        glyph_seed: u64,
        stroke_seed: u64,
        glyph: &str,
        is_left: bool,
    ) -> Vec<SiftFeature> {
        let mut keypoints = Vec::new();
        let chars: Vec<char> = glyph.chars().collect();

        for i in 0..KEYPOINTS_PER_EDGE {
            let base_seed = slip_seed.wrapping_add((i as u64) * 100);
            let x = self.pseudo_random(base_seed, 1);
            let y = self.pseudo_random(base_seed, 2);
            let scale = 0.5 + self.pseudo_random(glyph_seed, i as u64) * 2.0;
            let orientation = self.pseudo_random(stroke_seed, i as u64) * std::f64::consts::TAU;

            let mut descriptor = Vec::with_capacity(DESCRIPTOR_DIM);
            for j in 0..DESCRIPTOR_DIM {
                let char_idx = if is_left {
                    j.min(chars.len().saturating_sub(1))
                } else {
                    (chars.len() + j) % chars.len().max(1)
                };
                let ch = chars.get(char_idx).copied().unwrap_or(' ');
                let ch_val = (ch as u64) as f64 / 65536.0;
                let noise = self.pseudo_random(base_seed.wrapping_add(j as u64), 3);
                descriptor.push((ch_val * 0.7 + noise * 0.3).clamp(0.0, 1.0));
            }

            let norm: f64 = descriptor.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for d in descriptor.iter_mut() {
                    *d /= norm;
                }
            }

            keypoints.push(SiftFeature {
                keypoint_x: x,
                keypoint_y: y,
                scale,
                orientation,
                descriptor,
            });
        }

        keypoints
    }

    fn extract_stroke_continuation(&self, slip: &BambooSlip) -> Vec<f64> {
        let mut pattern = vec![0.0; 16];
        let remnant = &slip.remnant_strokes;

        let keywords_top = ["上", "首", "起", "始"];
        let keywords_bottom = ["下", "末", "尾", "终"];
        let keywords_left = ["左", "前"];
        let keywords_right = ["右", "后"];
        let keywords_partial = ["残", "缺", "损", "破", "断", "裂", "碎"];
        let keywords_connect = ["连", "接", "续", "贯", "通"];

        for (i, kw) in keywords_top.iter().enumerate() {
            if remnant.contains(kw) {
                pattern[i] += 0.8;
                pattern[i + 4] += 0.4;
            }
        }
        for (i, kw) in keywords_bottom.iter().enumerate() {
            if remnant.contains(kw) {
                pattern[i] += 0.7;
                pattern[i + 4] += 0.5;
            }
        }
        for (i, kw) in keywords_left.iter().enumerate() {
            if remnant.contains(kw) {
                pattern[i + 8] += 0.9;
            }
        }
        for (i, kw) in keywords_right.iter().enumerate() {
            if remnant.contains(kw) {
                pattern[i + 10] += 0.9;
            }
        }
        for kw in keywords_partial.iter() {
            if remnant.contains(kw) {
                for i in 0..12 {
                    pattern[i] += 0.1;
                }
            }
        }
        for kw in keywords_connect.iter() {
            if remnant.contains(kw) {
                pattern[12] += 0.6;
                pattern[13] += 0.4;
            }
        }

        let glyph_len = slip.glyph_features.chars().count();
        pattern[14] = (glyph_len as f64 / 20.0).clamp(0.0, 1.0);

        let tag_hash = self.hash_string(&slip.grammar_tag);
        pattern[15] = (tag_hash % 1000) as f64 / 1000.0;

        pattern
    }

    pub fn match_edges(
        &self,
        left: &EdgeFeatures,
        right: &EdgeFeatures,
    ) -> (f64, f64, usize) {
        let mut matched_pairs = Vec::new();
        let mut matched_indices = std::collections::HashSet::new();

        for (i, left_kp) in left.right_edge.iter().enumerate() {
            let mut best_match: Option<(usize, f64)> = None;
            for (j, right_kp) in right.left_edge.iter().enumerate() {
                if matched_indices.contains(&j) {
                    continue;
                }
                let sim = self.descriptor_similarity(&left_kp.descriptor, &right_kp.descriptor);
                let geo_score = self.geometry_compatibility(left_kp, right_kp);
                let combined = sim * 0.6 + geo_score * 0.4;

                if best_match.map_or(true, |(_, s)| combined > s) {
                    best_match = Some((j, combined));
                }
            }
            if let Some((j, score)) = best_match {
                matched_indices.insert(j);
                matched_pairs.push((i, j, score));
            }
        }

        matched_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        matched_pairs.truncate(KEYPOINTS_PER_EDGE.min(5));

        if matched_pairs.is_empty() {
            return (0.0, 0.0, 0);
        }

        let avg_similarity: f64 = matched_pairs.iter().map(|(_, _, s)| s).sum::<f64>() / matched_pairs.len() as f64;
        let geo_score = self.global_geometry_score(&matched_pairs, &left.right_edge, &right.left_edge);
        let count = matched_pairs.len();

        (avg_similarity, geo_score, count)
    }

    fn descriptor_similarity(&self, d1: &[f64], d2: &[f64]) -> f64 {
        let dot: f64 = d1.iter().zip(d2.iter()).map(|(a, b)| a * b).sum();
        dot.clamp(-1.0, 1.0).max(0.0)
    }

    fn geometry_compatibility(&self, kp1: &SiftFeature, kp2: &SiftFeature) -> f64 {
        let scale_diff = (kp1.scale - kp2.scale).abs();
        let scale_score = (-scale_diff * 2.0).exp();

        let mut orient_diff = (kp1.orientation - kp2.orientation).abs();
        if orient_diff > std::f64::consts::PI {
            orient_diff = std::f64::consts::TAU - orient_diff;
        }
        let orient_score = (-orient_diff * 1.5).exp();

        let pos_dist = ((kp1.keypoint_x - kp2.keypoint_x).powi(2)
            + (kp1.keypoint_y - kp2.keypoint_y).powi(2))
        .sqrt();
        let pos_score = (-pos_dist * 3.0).exp();

        scale_score * 0.4 + orient_score * 0.3 + pos_score * 0.3
    }

    fn global_geometry_score(
        &self,
        pairs: &[(usize, usize, f64)],
        left_keypoints: &[SiftFeature],
        right_keypoints: &[SiftFeature],
    ) -> f64 {
        if pairs.len() < 2 {
            return 0.5;
        }

        let mut total_transform_consistency = 0.0;
        let mut valid_pairs = 0;

        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                let (li1, ri1, _) = pairs[i];
                let (li2, ri2, _) = pairs[j];

                let lkp1 = &left_keypoints[li1];
                let lkp2 = &left_keypoints[li2];
                let rkp1 = &right_keypoints[ri1];
                let rkp2 = &right_keypoints[ri2];

                let l_dist = ((lkp1.keypoint_x - lkp2.keypoint_x).powi(2)
                    + (lkp1.keypoint_y - lkp2.keypoint_y).powi(2))
                .sqrt();
                let r_dist = ((rkp1.keypoint_x - rkp2.keypoint_x).powi(2)
                    + (rkp1.keypoint_y - rkp2.keypoint_y).powi(2))
                .sqrt();

                let dist_ratio = if l_dist > 0.0 && r_dist > 0.0 {
                    (l_dist / r_dist).min(r_dist / l_dist)
                } else {
                    0.5
                };

                let l_scale = (lkp1.scale - lkp2.scale).abs();
                let r_scale = (rkp1.scale - rkp2.scale).abs();
                let scale_consistency = 1.0 - (l_scale - r_scale).abs().min(1.0);

                let l_orient = (lkp1.orientation - lkp2.orientation).abs();
                let r_orient = (rkp1.orientation - rkp2.orientation).abs();
                let l_orient = if l_orient > std::f64::consts::PI {
                    std::f64::consts::TAU - l_orient
                } else {
                    l_orient
                };
                let r_orient = if r_orient > std::f64::consts::PI {
                    std::f64::consts::TAU - r_orient
                } else {
                    r_orient
                };
                let orient_consistency = 1.0 - (l_orient - r_orient).abs() / std::f64::consts::PI;

                let pair_consistency = dist_ratio * 0.4 + scale_consistency * 0.3 + orient_consistency * 0.3;
                total_transform_consistency += pair_consistency;
                valid_pairs += 1;
            }
        }

        if valid_pairs > 0 {
            total_transform_consistency / valid_pairs as f64
        } else {
            0.5
        }
    }

    pub fn stroke_continuation_similarity(&self, left: &EdgeFeatures, right: &EdgeFeatures) -> f64 {
        let v1 = &left.stroke_continuation;
        let v2 = &right.stroke_continuation;

        let mut dot = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..v1.len().min(v2.len()) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        let denom = norm1.sqrt() * norm2.sqrt();
        if denom > 0.0 {
            (dot / denom).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

impl Default for SiftMatcher {
    fn default() -> Self {
        Self::new()
    }
}