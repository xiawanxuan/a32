use crate::models::{BambooSlip, LinkScore, LinkResult, Weights};
use crate::distance::{jaro_winkler_similarity, GrammarDistance};
use std::collections::HashSet;

pub struct ScoringEngine {
    pub weights: Weights,
    pub prefix_weight: f64,
    grammar_distance: GrammarDistance,
}

impl ScoringEngine {
    pub fn new(weights: Weights) -> Self {
        ScoringEngine {
            weights,
            prefix_weight: 0.1,
            grammar_distance: GrammarDistance::new(),
        }
    }

    pub fn calculate_link_score(&self, from: &BambooSlip, to: &BambooSlip) -> LinkScore {
        let glyph_score = self.calculate_glyph_score(from, to);
        let stroke_score = self.calculate_stroke_score(from, to);
        let grammar_score = self.calculate_grammar_score(from, to);

        let total_score = glyph_score * self.weights.glyph_weight
            + stroke_score * self.weights.stroke_weight
            + grammar_score * self.weights.grammar_weight;

        LinkScore {
            from_id: from.slip_id.clone(),
            to_id: to.slip_id.clone(),
            total_score,
            glyph_score,
            stroke_score,
            grammar_score,
        }
    }

    fn calculate_glyph_score(&self, from: &BambooSlip, to: &BambooSlip) -> f64 {
        let from_end = from.glyph_features.chars().rev().take(4).collect::<String>();
        let to_start = to.glyph_features.chars().take(4).collect::<String>();
        let from_end_rev: String = from_end.chars().rev().collect();

        jaro_winkler_similarity(&from_end_rev, &to_start, self.prefix_weight)
    }

    fn calculate_stroke_score(&self, from: &BambooSlip, to: &BambooSlip) -> f64 {
        if from.remnant_strokes.is_empty() || to.remnant_strokes.is_empty() {
            return 0.5;
        }
        jaro_winkler_similarity(&from.remnant_strokes, &to.remnant_strokes, self.prefix_weight * 0.5)
    }

    fn calculate_grammar_score(&self, from: &BambooSlip, to: &BambooSlip) -> f64 {
        self.grammar_distance.similarity(&from.grammar_tag, &to.grammar_tag)
    }

    pub fn compute_all_pair_scores(&self, slips: &[BambooSlip]) -> Vec<LinkScore> {
        let mut scores = Vec::new();
        for i in 0..slips.len() {
            for j in 0..slips.len() {
                if i != j {
                    scores.push(self.calculate_link_score(&slips[i], &slips[j]));
                }
            }
        }
        scores
    }

    pub fn find_optimal_order(&self, slips: &[BambooSlip]) -> LinkResult {
        if slips.is_empty() {
            return LinkResult {
                order: Vec::new(),
                total_score: 0.0,
                link_scores: Vec::new(),
            };
        }

        let all_scores = self.compute_all_pair_scores(slips);
        let slip_map: std::collections::HashMap<&str, &BambooSlip> =
            slips.iter().map(|s| (s.slip_id.as_str(), s)).collect();

        let score_map: std::collections::HashMap<(String, String), &LinkScore> =
            all_scores.iter().map(|s| ((s.from_id.clone(), s.to_id.clone()), s)).collect();

        let mut best_result: Option<LinkResult> = None;

        for start_slip in slips {
            let result = self.greedy_path_from_start(start_slip, slips, &score_map, &slip_map);
            if best_result.is_none() || result.total_score > best_result.as_ref().unwrap().total_score {
                best_result = Some(result);
            }
        }

        best_result.unwrap_or_else(|| LinkResult {
            order: slips.to_vec(),
            total_score: 0.0,
            link_scores: Vec::new(),
        })
    }

    fn greedy_path_from_start(
        &self,
        start: &BambooSlip,
        slips: &[BambooSlip],
        score_map: &std::collections::HashMap<(String, String), &LinkScore>,
        _slip_map: &std::collections::HashMap<&str, &BambooSlip>,
    ) -> LinkResult {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<BambooSlip> = Vec::new();
        let mut link_scores: Vec<LinkScore> = Vec::new();
        let mut total_score = 0.0;

        let mut current = start;
        visited.insert(start.slip_id.clone());
        order.push(current.clone());

        while visited.len() < slips.len() {
            let mut best_next: Option<&BambooSlip> = None;
            let mut best_score: Option<&LinkScore> = None;

            for slip in slips {
                if visited.contains(&slip.slip_id) {
                    continue;
                }
                let key = (current.slip_id.clone(), slip.slip_id.clone());
                if let Some(score) = score_map.get(&key) {
                    if best_score.is_none() || score.total_score > best_score.unwrap().total_score {
                        best_score = Some(score);
                        best_next = Some(slip);
                    }
                }
            }

            if let Some(next_slip) = best_next {
                if let Some(score) = best_score {
                    link_scores.push((*score).clone());
                    total_score += score.total_score;
                }
                visited.insert(next_slip.slip_id.clone());
                order.push(next_slip.clone());
                current = next_slip;
            } else {
                break;
            }
        }

        LinkResult {
            order,
            total_score,
            link_scores,
        }
    }

    pub fn compute_order_score(&self, order: &[BambooSlip]) -> LinkResult {
        let mut link_scores = Vec::new();
        let mut total_score = 0.0;

        for i in 0..order.len().saturating_sub(1) {
            let score = self.calculate_link_score(&order[i], &order[i + 1]);
            total_score += score.total_score;
            link_scores.push(score);
        }

        LinkResult {
            order: order.to_vec(),
            total_score,
            link_scores,
        }
    }
}

impl Default for ScoringEngine {
    fn default() -> Self {
        ScoringEngine::new(Weights::default())
    }
}
