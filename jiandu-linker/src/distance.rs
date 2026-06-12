use std::collections::HashMap;

pub fn jaro_winkler_similarity(s1: &str, s2: &str, prefix_weight: f64) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_distance = ((len1.max(len2) / 2) - 1).max(0);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];

    let mut matches: f64 = 0.0;
    for i in 0..len1 {
        let start = if i > match_distance { i - match_distance } else { 0 };
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] {
                continue;
            }
            if s1_chars[i] == s2_chars[j] {
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1.0;
                break;
            }
        }
    }

    if matches == 0.0 {
        return 0.0;
    }

    let mut transpositions: f64 = 0.0;
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while k < len2 && !s2_matches[k] {
            k += 1;
        }
        if k < len2 && s1_chars[i] != s2_chars[k] {
            transpositions += 0.5;
        }
        k += 1;
    }

    let jaro = (matches / len1 as f64
        + matches / len2 as f64
        + (matches - transpositions) / matches)
        / 3.0;

    let mut prefix_length = 0;
    let min_len = len1.min(len2);
    for i in 0..min_len {
        if s1_chars[i] == s2_chars[i] {
            prefix_length += 1;
        } else {
            break;
        }
    }
    let prefix_length = prefix_length.min(4);

    jaro + (prefix_length as f64) * prefix_weight * (1.0 - jaro)
}

pub struct GrammarDistance {
    tag_categories: HashMap<String, usize>,
    category_distance: Vec<Vec<f64>>,
}

impl GrammarDistance {
    pub fn new() -> Self {
        let mut tag_categories = HashMap::new();
        tag_categories.insert("名词".to_string(), 0);
        tag_categories.insert("动词".to_string(), 1);
        tag_categories.insert("形容词".to_string(), 2);
        tag_categories.insert("副词".to_string(), 3);
        tag_categories.insert("介词".to_string(), 4);
        tag_categories.insert("连词".to_string(), 5);
        tag_categories.insert("代词".to_string(), 6);
        tag_categories.insert("数词".to_string(), 7);
        tag_categories.insert("量词".to_string(), 8);
        tag_categories.insert("助词".to_string(), 9);
        tag_categories.insert("语气词".to_string(), 10);

        let n = tag_categories.len();
        let mut category_distance = vec![vec![0.5; n]; n];

        for i in 0..n {
            category_distance[i][i] = 0.1;
        }

        category_distance[0][1] = 0.15;
        category_distance[1][0] = 0.2;

        category_distance[2][0] = 0.2;
        category_distance[0][2] = 0.3;

        category_distance[3][1] = 0.2;
        category_distance[1][3] = 0.3;

        category_distance[7][8] = 0.05;
        category_distance[8][7] = 0.15;

        category_distance[4][0] = 0.25;
        category_distance[0][4] = 0.35;

        category_distance[6][1] = 0.2;
        category_distance[1][6] = 0.3;

        category_distance[0][6] = 0.25;
        category_distance[6][0] = 0.25;

        category_distance[2][3] = 0.3;
        category_distance[3][2] = 0.35;

        category_distance[9][10] = 0.3;
        category_distance[10][9] = 0.35;

        GrammarDistance {
            tag_categories,
            category_distance,
        }
    }

    pub fn distance(&self, tag1: &str, tag2: &str) -> f64 {
        let cat1 = self.tag_categories.get(tag1).copied().unwrap_or(11);
        let cat2 = self.tag_categories.get(tag2).copied().unwrap_or(11);

        if cat1 >= self.category_distance.len() || cat2 >= self.category_distance.len() {
            return 0.5;
        }

        self.category_distance[cat1][cat2]
    }

    pub fn similarity(&self, tag1: &str, tag2: &str) -> f64 {
        1.0 - self.distance(tag1, tag2)
    }
}

impl Default for GrammarDistance {
    fn default() -> Self {
        Self::new()
    }
}
