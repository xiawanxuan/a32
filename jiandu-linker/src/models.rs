use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BambooSlip {
    #[serde(rename = "简号")]
    pub slip_id: String,

    #[serde(rename = "字形特征")]
    pub glyph_features: String,

    #[serde(rename = "残笔描述")]
    pub remnant_strokes: String,

    #[serde(rename = "语法标签")]
    pub grammar_tag: String,
}

#[derive(Debug, Clone)]
pub struct LinkScore {
    pub from_id: String,
    pub to_id: String,
    pub total_score: f64,
    pub glyph_score: f64,
    pub stroke_score: f64,
    pub grammar_score: f64,
}

#[derive(Debug, Clone)]
pub struct LinkResult {
    pub order: Vec<BambooSlip>,
    pub total_score: f64,
    pub link_scores: Vec<LinkScore>,
}

pub struct Weights {
    pub glyph_weight: f64,
    pub stroke_weight: f64,
    pub grammar_weight: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            glyph_weight: 0.4,
            stroke_weight: 0.3,
            grammar_weight: 0.3,
        }
    }
}
