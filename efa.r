library(readr)
library(psych)
library(EFA.dimensions)
library(corrplot)
library(EFAtools)

load_data <- function(filepath, drop_columns) {
  data <- read_csv(filepath)
  data <- data[, !(names(data) %in% drop_columns)]
  return(data)
}

preprocess_data <- function(data) {
  data[data == "N/A" | data == "-" | data == "NA" | data == ""] <- NA
  cat("Percent missing data:", round(sum(is.na(data)) / (nrow(data) * ncol(data)) * 100, 2), "%\n")
  data <- data.frame(lapply(data, function(x) ifelse(is.na(as.numeric(x)), mean(as.numeric(x), na.rm = TRUE), as.numeric(x))))

  # Remove subtests which are too hard in the HuggingFace Open LLM Leaderboard
  # data <- data[, colSums(data >= 26, na.rm = TRUE) >= 0.80 * nrow(data)]

  cat("Number of cases:", nrow(data), "\n")
  return(data)
}

# Calculates & prints skew and kurtosis
calculate_skew_kurt <- function(data) {
  univariate_skew <- c(min(skew(data)), max(skew(data)))
  univariate_kurt <- c(min(kurtosi(data)), max(kurtosi(data)))
  multivariate_kurt <- mardia(data, plot=FALSE)$kurtosis

  cat("Univariate Skewness:", univariate_skew[1], "-", univariate_skew[2], "\n")
  cat("Univariate Kurtosis:", univariate_kurt[1], "-", univariate_kurt[2], "\n")
  cat("Multivariate Kurtosis:", multivariate_kurt, "\n")
}

# Calculates & prints KMO
calculate_kmo <- function(data) {
  kmo_overall <- KMO(data)$KMO
  bart_results <- bartlett.test(data)

  cat("KMO Overall:", kmo_overall, "\n")
  cat("Bartlett's Test of Sphericity:", bart_results$statistic, bart_results$parameter, bart_results$p.value, "\n")
}

# Suggests how many factors to extract
do_extraction_analysis <- function(spearman_cor, n_cases) {
  unrotated_fa <- fa(r = spearman_cor, nfactors = 1, fm = "pa", rotate = "promax", max.iter = 100)
  g_var_accounted = unrotated_fa$Vaccounted[1, 1] / ncol(spearman_cor)
  cat("Percentage of variance accounted for by the first unrotated factor:", round(g_var_accounted * 100, 0), "%\n")
  print(PARALLEL(unrotated_fa$r, N=n_cases, verbose = FALSE))
  min_avg_partial <- MAP(unrotated_fa$r, corkind="spearman", Ncases=n_cases, verbose = FALSE)$NfactorsMAP
}

# Performs the factor analysis with the specified number of factors
do_factor_analysis <- function(spearman_cor, num_factors) {
  rotated_fa <- fa(r = spearman_cor, nfactors = num_factors, fm = "pa", rotate = "promax", max.iter = 100)
  fa_diagram <- fa.diagram(rotated_fa, digits = 3, sort = TRUE)
  resids = residuals(rotated_fa, diag=FALSE)
  resids <- resids[!is.na(resids)]
  residuals_histogram <- hist(resids, breaks=20, main="Residuals", xlab="Residuals")
  percentage_resids <- round(mean(abs(resids) > 0.05) * 100, 2)
  range_resids <- c(round(min(abs(resids)), 2), round(max(abs(resids)), 2))

  print(rotated_fa)
  cat("Percentage of residuals with absolute value greater than 0.05:", percentage_resids, "%\n")
  cat("Range of absolute residuals:", range_resids[1], "-", range_resids[2], "\n")
}

# Main function
perform_analysis <- function(filepath, drop_columns) {
  data <- load_data(filepath, drop_columns)
  data <- preprocess_data(data)
  
  spearman_cor <- cor(data, method = "spearman")
  cat("Average correlation:", mean(spearman_cor[upper.tri(spearman_cor, diag = FALSE)]), "\n")
  corrplot(spearman_cor, method = "color", type = "full", order = "hclust", tl.col = "black", tl.srt = 45, tl.cex = 1.5, addCoef.col = "white", addCoefasPercent = TRUE, number.cex = 1.1, cl.cex = 1.5)

  calculate_skew_kurt(data)
  calculate_kmo(data)
  do_extraction_analysis(spearman_cor, nrow(data))
  num_factors <- as.integer(readline(prompt = "Enter the number of factors to be analyzed based on the results of the extraction analysis: "))
  do_factor_analysis(spearman_cor, num_factors)
}

# "AX" is removed from GLUE because it's a diagnostic test. Leaving it in doesn't change the results, but it was removed for consistency.
perform_analysis('./data/hf_leaderboard.csv', c("model", "param_count", "AX"))