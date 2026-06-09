use crate::models::BambooSlip;
use crate::distance::jaro_winkler_similarity;

#[derive(Debug, Clone)]
pub struct EdgeDescriptor {
    pub chars: Vec<char>,
    pub stroke_pattern: Vec<u8>,
    pub direction: EdgeDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeDirection {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct FragmentMatch {
    pub left_id: String,
    pub right_id: String,
    pub confidence: f64,
    pub edge_similarity: f64,
    pub stroke_complement: f64,
    pub glyph_continuity: f64,
}

pub struct FragmentMatcher {
    pub edge_length: usize,
    pub prefix_weight: f64,
    pub edge_weight: f64,
    pub stroke_weight: f64,
    pub glyph_weight: f64,
}

impl Default for FragmentMatcher {
    fn default() -> Self {
        FragmentMatcher {
            edge_length: 3,
            prefix_weight: 0.1,
            edge_weight: 0.4,
            stroke_weight: 0.35,
            glyph_weight: 0.25,
        }
    }
}

impl FragmentMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extract_left_descriptor(&self, slip: &BambooSlip) -> EdgeDescriptor {
        let chars: Vec<char> = slip.glyph_features.chars().take(self.edge_length).collect();
        let stroke_pattern = self.extract_stroke_pattern(&slip.remnant_strokes, true);
        EdgeDescriptor {
            chars,
            stroke_pattern,
            direction: EdgeDirection::Left,
        }
    }

    pub fn extract_right_descriptor(&self, slip: &BambooSlip) -> EdgeDescriptor {
        let total = slip.glyph_features.chars().count();
        let start = total.saturating_sub(self.edge_length);
        let chars: Vec<char> = slip.glyph_features.chars().skip(start).collect();
        let stroke_pattern = self.extract_stroke_pattern(&slip.remnant_strokes, false);
        EdgeDescriptor {
            chars,
            stroke_pattern,
            direction: EdgeDirection::Right,
        }
    }

    fn extract_stroke_pattern(&self, remnant: &str, is_left: bool) -> Vec<u8> {
        if remnant.is_empty() {
            return vec![0; 8];
        }

        let mut pattern = vec![0u8; 8];
        let keywords_top = ["上", "首", "起", "始"];
        let keywords_bottom = ["下", "末", "尾", "终"];
        let keywords_left = ["左", "前"];
        let keywords_right = ["右", "后"];
        let keywords_partial = ["残", "缺", "损", "破", "断"];

        for kw in keywords_top.iter() {
            if remnant.contains(kw) {
                pattern[0] += 2;
                pattern[1] += 1;
            }
        }
        for kw in keywords_bottom.iter() {
            if remnant.contains(kw) {
                pattern[2] += 2;
                pattern[3] += 1;
            }
        }
        for kw in keywords_left.iter() {
            if remnant.contains(kw) {
                pattern[4] += 2;
                pattern[5] += 1;
            }
        }
        for kw in keywords_right.iter() {
            if remnant.contains(kw) {
                pattern[6] += 2;
                pattern[7] += 1;
            }
        }
        for kw in keywords_partial.iter() {
            if remnant.contains(kw) {
                for i in 0..8 {
                    pattern[i] = pattern[i].saturating_add(1);
                }
            }
        }

        if is_left {
            pattern[4] = pattern[4].saturating_add(1);
        } else {
            pattern[6] = pattern[6].saturating_add(1);
        }

        pattern
    }

    pub fn match_fragments(&self, left: &BambooSlip, right: &BambooSlip) -> FragmentMatch {
        let right_desc = self.extract_right_descriptor(left);
        let left_desc = self.extract_left_descriptor(right);

        let edge_similarity = self.calculate_edge_similarity(&right_desc, &left_desc);
        let stroke_complement = self.calculate_stroke_complementarity(&right_desc, &left_desc);
        let glyph_continuity = self.calculate_glyph_continuity(left, right);

        let confidence = edge_similarity * self.edge_weight
            + stroke_complement * self.stroke_weight
            + glyph_continuity * self.glyph_weight;

        FragmentMatch {
            left_id: left.slip_id.clone(),
            right_id: right.slip_id.clone(),
            confidence,
            edge_similarity,
            stroke_complement,
            glyph_continuity,
        }
    }

    fn calculate_edge_similarity(&self, left_edge: &EdgeDescriptor, right_edge: &EdgeDescriptor) -> f64 {
        let left_str: String = left_edge.chars.iter().collect();
        let right_str: String = right_edge.chars.iter().collect();
        jaro_winkler_similarity(&left_str, &right_str, self.prefix_weight)
    }

    fn calculate_stroke_complementarity(&self, left_edge: &EdgeDescriptor, right_edge: &EdgeDescriptor) -> f64 {
        if left_edge.stroke_pattern.len() != right_edge.stroke_pattern.len() {
            return 0.5;
        }

        let n = left_edge.stroke_pattern.len();
        let mut complement_score = 0.0;
        let mut max_possible = 0.0;

        for i in 0..n {
            let l = left_edge.stroke_pattern[i] as f64;
            let r = right_edge.stroke_pattern[i] as f64;

            let sum = l + r;
            let diff = (l - r).abs();
            let complement = if sum > 0.0 {
                1.0 - (diff / sum)
            } else {
                0.5
            };

            complement_score += complement * (l.max(r));
            max_possible += l.max(r).max(1.0);
        }

        if max_possible > 0.0 {
            complement_score / max_possible
        } else {
            0.5
        }
    }

    fn calculate_glyph_continuity(&self, left: &BambooSlip, right: &BambooSlip) -> f64 {
        let left_all: Vec<char> = left.glyph_features.chars().collect();
        let right_all: Vec<char> = right.glyph_features.chars().collect();

        if left_all.is_empty() || right_all.is_empty() {
            return 0.5;
        }

        let mut max_overlap = 0;
        let max_check = self.edge_length.min(left_all.len()).min(right_all.len());

        for overlap in 1..=max_check {
            let left_suffix: String = left_all.iter().rev().take(overlap).rev().copied().collect();
            let right_prefix: String = right_all.iter().take(overlap).copied().collect();

            if left_suffix == right_prefix {
                max_overlap = overlap;
            }
        }

        if max_check > 0 {
            max_overlap as f64 / max_check as f64
        } else {
            0.5
        }
    }

    pub fn find_all_matches(&self, slips: &[BambooSlip]) -> Vec<FragmentMatch> {
        let mut matches = Vec::new();

        for i in 0..slips.len() {
            for j in 0..slips.len() {
                if i != j {
                    let m = self.match_fragments(&slips[i], &slips[j]);
                    matches.push(m);
                }
            }
        }

        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    pub fn find_top_matches(&self, slips: &[BambooSlip], top_n: usize) -> Vec<FragmentMatch> {
        let mut matches = self.find_all_matches(slips);
        matches.truncate(top_n);
        matches
    }
}
