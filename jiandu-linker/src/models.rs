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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct SiftFeature {
    pub keypoint_x: f64,
    pub keypoint_y: f64,
    pub scale: f64,
    pub orientation: f64,
    pub descriptor: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    pub slip_id: String,
    pub left_edge: Vec<SiftFeature>,
    pub right_edge: Vec<SiftFeature>,
    pub stroke_continuation: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FragmentMatch {
    pub left_id: String,
    pub right_id: String,
    pub confidence: f64,
    pub sift_similarity: f64,
    pub edge_geometry_score: f64,
    pub stroke_continuity: f64,
    pub glyph_overlap_score: f64,
    pub matched_keypoints: usize,
}

#[derive(Debug, Clone)]
pub struct FragmentRecommendation {
    pub matches: Vec<FragmentMatch>,
    pub recommended_merges: Vec<(String, String, f64)>,
}

#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub show_scores: bool,
    pub show_details: bool,
    pub indent_width: usize,
    pub max_width: usize,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        VisualizationConfig {
            show_scores: true,
            show_details: true,
            indent_width: 4,
            max_width: 100,
        }
    }
}
