#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:31:42 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================
# LDA TOPIC MODELLING FOR UNGA CLIMATE DISCOURSE
# =====================================================

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# =====================================================
# 1. LOAD DATA
# =====================================================

file_path = r"/Users/user/Desktop/Other Docs/UN/LDA Analysis /Raw UNGA Data.xlsx"
df = pd.read_excel(file_path)

print("Columns in dataset:")
print(df.columns)


# =====================================================
# 2. PREPROCESS TEXT
# =====================================================

stop_words = set(stopwords.words("english"))

custom_stopwords = {
    "united", "nations", "general", "assembly", "session",
    "president", "mr", "madam", "world", "international",
    "country", "countries", "people", "must", "also",
    "would", "could", "shall", "today", "year", "years"
}

stop_words.update(custom_stopwords)

def preprocess(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = word_tokenize(text)
    
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 3
    ]
    
    return tokens

df["tokens"] = df["Cleaned_Speech_Text"].apply(preprocess)
df = df[df["tokens"].map(len) > 0].copy()

# =====================================================
# 3. CREATE DICTIONARY AND CORPUS
# =====================================================

dictionary = corpora.Dictionary(df["tokens"])
dictionary.filter_extremes(no_below=10, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in df["tokens"]]

# =====================================================
# 4. FIND OPTIMAL K: COHERENCE, PERPLEXITY, AND ELBOW
# =====================================================

k_range = range(5, 31)

coherence_scores = []
perplexity_scores = []

for k in k_range:
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=42,
        passes=10
    )
    
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=df["tokens"],
        dictionary=dictionary,
        coherence="c_v"
    )
    
    coherence = coherence_model.get_coherence()
    perplexity = lda_model.log_perplexity(corpus)
    
    coherence_scores.append(coherence)
    perplexity_scores.append(perplexity)
    
    print(f"K={k} | Coherence={coherence:.4f} | Log Perplexity={perplexity:.4f}")

k_results_df = pd.DataFrame({
    "K": list(k_range),
    "Coherence": coherence_scores,
    "Log_Perplexity": perplexity_scores
})

# Coherence-based optimal K
optimal_index = np.argmax(coherence_scores)
optimal_k = list(k_range)[optimal_index]

# Elbow method based on coherence curve
x = np.array(list(k_range))
y = np.array(coherence_scores)

point_start = np.array([x[0], y[0]])
point_end = np.array([x[-1], y[-1]])

distances = []

for i in range(len(x)):
    point = np.array([x[i], y[i]])
    distance = np.abs(
        np.cross(point_end - point_start, point_start - point)
    ) / np.linalg.norm(point_end - point_start)
    distances.append(distance)

elbow_index = np.argmax(distances)
elbow_k = list(k_range)[elbow_index]

k_results_df["Elbow_Distance"] = distances

print("\n====================================")
print("OPTIMAL K RESULTS")
print("====================================")
print(f"Optimal K based on coherence: {optimal_k}")
print(f"Coherence score: {coherence_scores[optimal_index]:.4f}")
print(f"Log perplexity: {perplexity_scores[optimal_index]:.4f}")
print(f"Elbow K based on coherence curve: {elbow_k}")

# =====================================================
# 5. PLOT K SELECTION
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(k_range, coherence_scores, marker="o")
plt.axvline(optimal_k, linestyle="--", label=f"Best coherence K={optimal_k}")
plt.axvline(elbow_k, linestyle=":", label=f"Elbow K={elbow_k}")
plt.title("Coherence Score vs Number of Topics")
plt.xlabel("Number of Topics (K)")
plt.ylabel("Coherence Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(k_range, perplexity_scores, marker="o")
plt.title("Log Perplexity vs Number of Topics")
plt.xlabel("Number of Topics (K)")
plt.ylabel("Log Perplexity")
plt.grid(True)
plt.show()

# =====================================================
# 6. TRAIN FINAL LDA MODEL
# =====================================================

final_lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=optimal_k,
    random_state=42,
    passes=20
)

# =====================================================
# 7. EXTRACT TOPIC TERMS
# =====================================================

topic_rows = []

for idx in range(optimal_k):
    terms = final_lda.show_topic(idx, topn=15)
    
    print(f"\nTopic {idx}:")
    print(terms)
    
    for rank, (term, weight) in enumerate(terms, start=1):
        topic_rows.append({
            "Topic": idx,
            "Rank": rank,
            "Term": term,
            "Weight": weight
        })

topic_terms_df = pd.DataFrame(topic_rows)

# =====================================================
# 8. ASSIGN DOMINANT TOPIC
# =====================================================

def get_dominant_topic_and_prob(bow):
    topics = final_lda.get_document_topics(bow)
    if topics:
        dominant_topic, probability = max(topics, key=lambda x: x[1])
        return dominant_topic, probability
    return None, None

dominant_results = [get_dominant_topic_and_prob(bow) for bow in corpus]

df["Dominant_Topic"] = [x[0] for x in dominant_results]
df["Dominant_Topic_Probability"] = [x[1] for x in dominant_results]

# =====================================================
# 9. TOPIC SCORES FOR EACH DOCUMENT
# =====================================================

topic_distributions = []

for bow in corpus:
    topic_probs = final_lda.get_document_topics(
        bow,
        minimum_probability=0
    )
    
    topic_distributions.append({
        f"Topic_{topic_id}_Score": prob
        for topic_id, prob in topic_probs
    })

topic_scores_df = pd.DataFrame(topic_distributions)

df_with_topic_scores = pd.concat(
    [df.reset_index(drop=True), topic_scores_df.reset_index(drop=True)],
    axis=1
)

# =====================================================
# 10. TOPIC DOMINANCE ACROSS YEARS AND CONTINENTS
# =====================================================

topic_by_year_counts = pd.crosstab(
    df["Year"],
    df["Dominant_Topic"]
)

topic_by_year_percent = pd.crosstab(
    df["Year"],
    df["Dominant_Topic"],
    normalize="index"
) * 100

topic_by_continent_counts = pd.crosstab(
    df["Continent"],
    df["Dominant_Topic"]
)

topic_by_continent_percent = pd.crosstab(
    df["Continent"],
    df["Dominant_Topic"],
    normalize="index"
) * 100

# =====================================================
# 11. ASSIGN TOPICS TO SIX THEMES
# =====================================================

theme_map = {
    0: "Global Climate Governance and Multilateral Processes",
    4: "Global Climate Governance and Multilateral Processes",

    1: "Environmental Conventions and Ecological Change",

    2: "Small Island States and Existential Risk",

    3: "Climate Impacts, Vulnerability, and Development",
    6: "Climate Impacts, Vulnerability, and Development",
    9: "Climate Impacts, Vulnerability, and Development",

    5: "Mitigation, Emissions, and Carbon Reduction",
    8: "Mitigation, Emissions, and Carbon Reduction",

    7: "Energy Transition and Green Economy"
}

df["Theme"] = df["Dominant_Topic"].map(theme_map)

unmapped_topics = df[df["Theme"].isna()]["Dominant_Topic"].unique()

print("\nUnmapped topics:")
print(unmapped_topics)

theme_by_continent = pd.crosstab(
    df["Continent"],
    df["Theme"],
    normalize="index"
) * 100

theme_by_year = pd.crosstab(
    df["Year"],
    df["Theme"],
    normalize="index"
) * 100

# =====================================================
# 12. CORPUS-LEVEL TOPIC PERCENTAGES
# =====================================================

num_topics = final_lda.num_topics
topic_totals = np.zeros(num_topics)

for bow in corpus:
    topic_probs = final_lda.get_document_topics(bow, minimum_probability=0)
    
    for topic_id, prob in topic_probs:
        topic_totals[topic_id] += prob

topic_percentages = topic_totals / topic_totals.sum() * 100

topic_distribution_df = pd.DataFrame({
    "Topic": range(num_topics),
    "Percentage": topic_percentages
}).sort_values(by="Percentage", ascending=False)

# =====================================================
# 13. PLOTS
# =====================================================

topic_by_year_percent.plot(
    kind="line",
    figsize=(12, 6),
    marker="o"
)

plt.title("Topic Dominance Across Years")
plt.xlabel("Year")
plt.ylabel("Percentage of Speeches")
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

topic_by_continent_percent.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Topic Dominance Across Continents")
plt.xlabel("Continent")
plt.ylabel("Percentage of Speeches")
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

theme_by_continent.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Distribution of LDA Themes Across Continents")
plt.xlabel("Continent")
plt.ylabel("Percentage of Speeches")
plt.xticks(rotation=45, ha="right")
plt.legend(
    title="Theme",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()


# =====================================================
# 13B. THEME DISTRIBUTION OVER YEARS + TREND ANALYSIS
# =====================================================

from scipy.stats import linregress

# Ensure Year is numeric
theme_by_year_trend = theme_by_year.copy()
theme_by_year_trend.index = theme_by_year_trend.index.astype(int)

# -------------------------------
# Plot theme distribution over time
# -------------------------------

plt.figure(figsize=(14, 7))

for theme in theme_by_year_trend.columns:
    plt.plot(
        theme_by_year_trend.index,
        theme_by_year_trend[theme],
        marker="o",
        label=theme
    )

plt.title("Distribution of Themes Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage of Speeches")
plt.legend(
    title="Theme",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Trend analysis for each theme
# -------------------------------

theme_trend_results = []

x = theme_by_year_trend.index.values

for theme in theme_by_year_trend.columns:
    y = theme_by_year_trend[theme].values
    
    slope, intercept, r_value, p_value, std_error = linregress(x, y)
    
    theme_trend_results.append({
        "Theme": theme,
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_value ** 2,
        "p_value": p_value,
        "Std_Error": std_error
    })

theme_trend_results_df = pd.DataFrame(theme_trend_results)

print("\n====================================")
print("THEME TREND ANALYSIS RESULTS")
print("====================================")
print(theme_trend_results_df)

# -------------------------------
# Optional: plot individual trend lines
# -------------------------------

for theme in theme_by_year_trend.columns:
    x = theme_by_year_trend.index.values
    y = theme_by_year_trend[theme].values
    
    slope, intercept, r_value, p_value, std_error = linregress(x, y)
    trend_line = intercept + slope * x
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", label="Observed")
    plt.plot(x, trend_line, linestyle="--", label="Trend line")
    
    plt.title(f"Temporal Trend: {theme}")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Speeches")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================
# 13C. CHI-SQUARE TEST: THEMES BY CONTINENT
#      WITH POST HOC TESTS AND EFFECT SIZE
# =====================================================

from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# Create contingency table: Continent x Theme
theme_continent_counts = pd.crosstab(
    df["Continent"],
    df["Theme"]
)

# Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(theme_continent_counts)

n = theme_continent_counts.to_numpy().sum()
rows, cols = theme_continent_counts.shape

# Cramer's V effect size
cramers_v = np.sqrt(
    chi2 / (n * (min(rows - 1, cols - 1)))
)

chi_square_results_df = pd.DataFrame({
    "Test": ["Chi-square test of independence"],
    "Chi_square": [chi2],
    "df": [dof],
    "p_value": [p],
    "N": [n],
    "Cramers_V": [cramers_v]
})

print("\n====================================")
print("CHI-SQUARE TEST: THEME BY CONTINENT")
print("====================================")
print(chi_square_results_df)

# Expected frequencies table
expected_df = pd.DataFrame(
    expected,
    index=theme_continent_counts.index,
    columns=theme_continent_counts.columns
)

# =====================================================
# Post hoc analysis using adjusted standardized residuals
# =====================================================

observed = theme_continent_counts.to_numpy()

row_totals = observed.sum(axis=1, keepdims=True)
col_totals = observed.sum(axis=0, keepdims=True)

row_props = row_totals / n
col_props = col_totals / n

# Adjusted standardized residuals
adjusted_residuals = (observed - expected) / np.sqrt(
    expected * (1 - row_props) * (1 - col_props)
)

adjusted_residuals_df = pd.DataFrame(
    adjusted_residuals,
    index=theme_continent_counts.index,
    columns=theme_continent_counts.columns
)

# Convert residuals to p-values
from scipy.stats import norm

posthoc_rows = []

for i, continent in enumerate(theme_continent_counts.index):
    for j, theme in enumerate(theme_continent_counts.columns):
        residual = adjusted_residuals[i, j]
        raw_p = 2 * (1 - norm.cdf(abs(residual)))
        
        posthoc_rows.append({
            "Continent": continent,
            "Theme": theme,
            "Observed_Count": observed[i, j],
            "Expected_Count": expected[i, j],
            "Adjusted_Residual": residual,
            "Raw_p": raw_p
        })

posthoc_df = pd.DataFrame(posthoc_rows)

# Bonferroni correction for multiple comparisons
posthoc_df["Bonferroni_p"] = multipletests(
    posthoc_df["Raw_p"],
    method="bonferroni"
)[1]

posthoc_df["Significant_Bonferroni"] = posthoc_df["Bonferroni_p"] < 0.05

# Add direction of difference
posthoc_df["Direction"] = np.where(
    posthoc_df["Adjusted_Residual"] > 0,
    "Higher than expected",
    "Lower than expected"
)

print("\n====================================")
print("POST HOC ANALYSIS: ADJUSTED RESIDUALS")
print("====================================")
print(posthoc_df)

# Filter only significant post hoc results
significant_posthoc_df = posthoc_df[
    posthoc_df["Significant_Bonferroni"] == True
].copy()

print("\n====================================")
print("SIGNIFICANT POST HOC RESULTS")
print("====================================")
print(significant_posthoc_df)

# =====================================================
# SAVE CHI-SQUARE RESULTS TO A SEPARATE EXCEL FILE
# =====================================================

chi_output_file = os.path.join(output_folder, "UNGA_ChiSquare_Results.xlsx")

with pd.ExcelWriter(chi_output_file, engine="openpyxl") as writer:
    theme_continent_counts.to_excel(writer, sheet_name="Observed_Counts")
    expected_df.to_excel(writer, sheet_name="Expected_Counts")
    adjusted_residuals_df.to_excel(writer, sheet_name="Adjusted_Residuals")
    chi_square_results_df.to_excel(writer, sheet_name="Chi_Square_Results", index=False)
    posthoc_df.to_excel(writer, sheet_name="Posthoc_All", index=False)
    significant_posthoc_df.to_excel(writer, sheet_name="Posthoc_Significant", index=False)

print(f"\nChi-square results saved to: {chi_output_file}")



# =====================================================
# 13D. EXTRACT 10 ILLUSTRATIVE EXCERPTS BY THEME
# =====================================================

import random
import nltk

nltk.download("punkt")

random.seed(42)

required_cols = ["Theme", "Year", "Country", "Post", "Cleaned_Speech_Text"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

def get_excerpt(text):
    text = str(text)
    sentences = nltk.sent_tokenize(text)
    return sentences[0] if sentences else text[:200]

excerpt_rows = []

themes = df["Theme"].dropna().unique()

for theme in themes:
    theme_df = df[df["Theme"] == theme].copy()
    theme_df = theme_df.dropna(subset=["Year", "Country", "Post", "Cleaned_Speech_Text"])
    
    # Shuffle for variety but keep reproducibility
    theme_df = theme_df.sample(frac=1, random_state=42).copy()
    
    selected = []
    used_years = set()
    
    # First, prioritise excerpts from different years
    for _, row in theme_df.iterrows():
        if row["Year"] not in used_years:
            selected.append(row)
            used_years.add(row["Year"])
        if len(selected) == 10:
            break
    
    # If fewer than 10 unique years, fill remaining slots randomly
    if len(selected) < 10:
        selected_indices = [row.name for row in selected]
        remaining_df = theme_df.drop(index=selected_indices, errors="ignore")
        
        additional_needed = min(10 - len(selected), len(remaining_df))
        
        if additional_needed > 0:
            additional = remaining_df.sample(
                n=additional_needed,
                random_state=42
            )
            selected.extend([row for _, row in additional.iterrows()])
    
    for row in selected:
        excerpt_rows.append({
            "Theme": theme,
            "Post": row["Post"],
            "Country": row["Country"],
            "Year": int(row["Year"]),
            "Excerpt": get_excerpt(row["Cleaned_Speech_Text"])
        })

excerpts_df = pd.DataFrame(excerpt_rows)

excerpts_df = excerpts_df.sort_values(
    by=["Theme", "Year"]
).reset_index(drop=True)

print("\n====================================")
print("10 ILLUSTRATIVE EXCERPTS BY THEME")
print("====================================")
print(excerpts_df)

excerpt_file = os.path.join(output_folder, "Theme_Excerpts_10_Per_Theme.xlsx")

excerpts_df.to_excel(excerpt_file, index=False)

print(f"\nExcerpts saved to: {excerpt_file}")




# =====================================================
# 14. SAVE ALL OUTPUTS INTO ONE EXCEL FILE
# =====================================================

output_folder = r"/Users/user/Desktop/Other Docs/UN/Health and climate change/LDA_Results"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "UNGA_LDA_All_Results.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Dominant_Topics", index=False)
    df_with_topic_scores.to_excel(writer, sheet_name="Document_Topic_Scores", index=False)
    topic_terms_df.to_excel(writer, sheet_name="Topic_Terms", index=False)
    k_results_df.to_excel(writer, sheet_name="K_Selection", index=False)
    topic_distribution_df.to_excel(writer, sheet_name="Corpus_Topic_Distribution", index=False)
    topic_by_year_counts.to_excel(writer, sheet_name="Topic_Year_Counts")
    topic_by_year_percent.to_excel(writer, sheet_name="Topic_Year_Percent")
    topic_by_continent_counts.to_excel(writer, sheet_name="Topic_Continent_Counts")
    topic_by_continent_percent.to_excel(writer, sheet_name="Topic_Continent_Percent")
    theme_by_continent.to_excel(writer, sheet_name="Theme_Continent_Percent")
    theme_by_year.to_excel(writer, sheet_name="Theme_Year_Percent")
    theme_trend_results_df.to_excel(writer, sheet_name="Theme_Trend_Analysis", index=False)
    theme_continent_counts.to_excel(writer, sheet_name="Theme_Continent_Counts")
expected_df.to_excel(writer, sheet_name="Expected_Frequencies")
adjusted_residuals_df.to_excel(writer, sheet_name="Adjusted_Residuals")
chi_square_results_df.to_excel(writer, sheet_name="Chi_Square_Results", index=False)
posthoc_df.to_excel(writer, sheet_name="Posthoc_All", index=False)
significant_posthoc_df.to_excel(writer, sheet_name="Posthoc_Significant", index=False)

print("\nLDA analysis complete.")
print(f"All outputs saved in one Excel file: {output_file}")