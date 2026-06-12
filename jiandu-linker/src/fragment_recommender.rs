use crate::models::{BambooSlip, FragmentMatch, FragmentRecommendation, EdgeFeatures};
use crate::sift_matching::SiftMatcher;
use crate::distance::jaro_winkler_similarity;
use std::collections::{HashMap, HashSet};

pub struct FragmentRecommender {
    sift_matcher: SiftMatcher,
    pub sift_weight: f64,
    pub geometry_weight: f64,
    pub stroke_weight: f64,
    pub glyph_weight: f64,
    pub confidence_threshold: f64,
}

impl Default for FragmentRecommender {
    fn default() -> Self {
        FragmentRecommender {
            sift_matcher: SiftMatcher::new(),
            sift_weight: 0.35,
            geometry_weight: 0.25,
            stroke_weight: 0.2,
            glyph_weight: 0.2,
            confidence_threshold: 0.6,
        }
    }
}

impl FragmentRecommender {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn precompute_features(&mut self, slips: &[BambooSlip]) -> HashMap<String, EdgeFeatures> {
        let mut features = HashMap::new();
        for slip in slips {
            let feat = self.sift_matcher.extract_sift_features(slip);
            features.insert(slip.slip_id.clone(), feat);
        }
        features
    }

    pub fn recommend_matches(
        &mut self,
        slips: &[BambooSlip],
    ) -> FragmentRecommendation {
        let features = self.precompute_features(slips);
        let mut matches = Vec::new();

        for i in 0..slips.len() {
            for j in 0..slips.len() {
                if i == j {
                    continue;
                }

                let left_slip = &slips[i];
                let right_slip = &slips[j];

                let left_feat = features.get(&left_slip.slip_id).unwrap();
                let right_feat = features.get(&right_slip.slip_id).unwrap();

                let (sift_sim, geometry_score, matched_kps) = self.sift_matcher
                    .match_edges(left_feat, right_feat);

                let stroke_cont = self.sift_matcher
                    .stroke_continuation_similarity(left_feat, right_feat);

                let glyph_overlap = self.calculate_glyph_overlap(left_slip, right_slip);

                let confidence = sift_sim * self.sift_weight
                    + geometry_score * self.geometry_weight
                    + stroke_cont * self.stroke_weight
                    + glyph_overlap * self.glyph_weight;

                if confidence >= self.confidence_threshold {
                    matches.push(FragmentMatch {
                        left_id: left_slip.slip_id.clone(),
                        right_id: right_slip.slip_id.clone(),
                        confidence,
                        sift_similarity: sift_sim,
                        edge_geometry_score: geometry_score,
                        stroke_continuity: stroke_cont,
                        glyph_overlap_score: glyph_overlap,
                        matched_keypoints: matched_kps,
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        let recommended_merges = self.generate_merge_recommendations(&matches, slips.len());

        FragmentRecommendation {
            matches,
            recommended_merges,
        }
    }

    fn calculate_glyph_overlap(&self, left: &BambooSlip, right: &BambooSlip) -> f64 {
        let left_chars: Vec<char> = left.glyph_features.chars().collect();
        let right_chars: Vec<char> = right.glyph_features.chars().collect();

        if left_chars.is_empty() || right_chars.is_empty() {
            return 0.5;
        }

        let max_overlap = left_chars.len().min(right_chars.len()).min(4);
        let mut best_overlap = 0;

        for overlap in 1..=max_overlap {
            let left_suffix: String = left_chars
                .iter()
                .rev()
                .take(overlap)
                .rev()
                .copied()
                .collect();
            let right_prefix: String = right_chars
                .iter()
                .take(overlap)
                .copied()
                .collect();

            let sim = jaro_winkler_similarity(&left_suffix, &right_prefix, 0.15);
            if sim >= 0.7 {
                best_overlap = overlap;
            }
        }

        let left_rem = left.remnant_strokes.to_lowercase();
        let right_rem = right.remnant_strokes.to_lowercase();

        let left_has_right = left_rem.contains("右") || left_rem.contains("末") || left_rem.contains("尾");
        let right_has_left = right_rem.contains("左") || right_rem.contains("首") || right_rem.contains("起");

        let stroke_bonus = if left_has_right && right_has_left {
            0.15
        } else if left_has_right || right_has_left {
            0.05
        } else {
            0.0
        };

        (best_overlap as f64 / max_overlap as f64) * 0.85 + stroke_bonus
    }

    fn generate_merge_recommendations(
        &self,
        matches: &[FragmentMatch],
        total_slips: usize,
    ) -> Vec<(String, String, f64)> {
        let mut used_ids = HashSet::new();
        let mut recommendations = Vec::new();
        let max_recommendations = (total_slips / 2).max(1).min(matches.len());

        for m in matches {
            if used_ids.contains(&m.left_id) || used_ids.contains(&m.right_id) {
                continue;
            }

            let has_reverse = matches.iter().any(|other| {
                other.left_id == m.right_id && other.right_id == m.left_id
            });

            if has_reverse {
                let reverse_match = matches.iter().find(|other| {
                    other.left_id == m.right_id && other.right_id == m.left_id
                }).unwrap();
                if m.confidence < reverse_match.confidence {
                    continue;
                }
            }

            if recommendations.len() >= max_recommendations {
                break;
            }

            used_ids.insert(m.left_id.clone());
            used_ids.insert(m.right_id.clone());
            recommendations.push((m.left_id.clone(), m.right_id.clone(), m.confidence));
        }

        recommendations
    }

    pub fn set_weights(
        &mut self,
        sift: f64,
        geometry: f64,
        stroke: f64,
        glyph: f64,
    ) {
        let total = sift + geometry + stroke + glyph;
        if total > 0.0 {
            self.sift_weight = sift / total;
            self.geometry_weight = geometry / total;
            self.stroke_weight = stroke / total;
            self.glyph_weight = glyph / total;
        }
    }

    pub fn get_confidence_level(confidence: f64) -> &'static str {
        if confidence >= 0.9 {
            "极高"
        } else if confidence >= 0.8 {
            "很高"
        } else if confidence >= 0.7 {
            "较高"
        } else if confidence >= 0.6 {
            "中等"
        } else {
            "较低"
        }
    }
}
