# Takes opportunity tables and runs the stats model on each one.
# Fits a negative binomial regression model that accounts for mutation class, species, and the number of opportunities as an offset.
# Contrast vectors calculate incidence rate ratios comparing groups of interest (e.g. comparing allele frequency classes to each other).
# P-values are calculated using Wald Chi-Squared tests and then adjusted for multiple testing using the Holm method.
# Results include effect sizes, confidence intervals, and significance labels (CSV written out for each).

library(MASS)
library(dplyr)

# Function definition

combined_nb_contrasts <- function(file_path,
                                  out_file,
                                  group_var,
                                  group_levels,
                                  mut_var           = "mut_class",
                                  species_var       = "species",
                                  response_var      = "n_mut",
                                  opp_var           = "opportunities",
                                  pairwise          = TRUE,
                                  adjust_method     = "holm",
                                  extra_comparisons = NULL) {

  if (is.character(file_path)) {
    df <- read.csv(file_path, stringsAsFactors = FALSE)
  } else {
    df <- file_path
  }

  df[[mut_var]]     <- factor(df[[mut_var]])
  df[[species_var]] <- factor(df[[species_var]])
  df[[group_var]]   <- factor(df[[group_var]], levels = group_levels)

  df <- df %>% filter(.data[[opp_var]] > 0)

  form <- as.formula(
    paste0(
      response_var, " ~ ",
      mut_var, " * ", group_var, " + ",
      species_var, " + offset(log(", opp_var, "))"
    )
  )

  fit <- glm.nb(form, data = df)

  coef_vec   <- coef(fit)
  vc         <- vcov(fit)
  coef_names <- names(coef_vec)

  sig_label <- function(p) {
    if (is.na(p))  return(NA_character_)
    if (p < 0.001) return("***")
    if (p < 0.01)  return("**")
    if (p < 0.05)  return("*")
    return("ns")
  }

  get_main_term <- function(level) {
    if (level == group_levels[1]) return(NA_character_)
    cand <- paste0(group_var, level)
    if (cand %in% coef_names) return(cand)
    return(NA_character_)
  }

  get_interaction_term <- function(mut_level, group_level) {
    if (group_level == group_levels[1]) return(NA_character_)
    cand1 <- paste0(mut_var, mut_level, ":", group_var, group_level)
    cand2 <- paste0(group_var, group_level, ":", mut_var, mut_level)
    if (cand1 %in% coef_names) return(cand1)
    if (cand2 %in% coef_names) return(cand2)
    return(NA_character_)
  }

  # Standard comparisons
  if (pairwise) {
    comp_grid <- t(combn(group_levels, 2))
    comp_df   <- data.frame(
      level_a = comp_grid[, 1],
      level_b = comp_grid[, 2],
      stringsAsFactors = FALSE
    )
  } else {
    comp_df <- data.frame(
      level_a = group_levels[-1],
      level_b = group_levels[1],
      stringsAsFactors = FALSE
    )
  }

  mut_levels <- levels(df[[mut_var]])
  results    <- list()
  k          <- 1

  for (mc in mut_levels) {
    for (i in seq_len(nrow(comp_df))) {

      a <- comp_df$level_a[i]
      b <- comp_df$level_b[i]

      L <- rep(0, length(coef_vec))
      names(L) <- coef_names

      main_a <- get_main_term(a)
      main_b <- get_main_term(b)
      if (!is.na(main_a)) L[main_a] <- L[main_a] + 1
      if (!is.na(main_b)) L[main_b] <- L[main_b] - 1

      int_a <- get_interaction_term(mc, a)
      int_b <- get_interaction_term(mc, b)
      if (!is.na(int_a)) L[int_a] <- L[int_a] + 1
      if (!is.na(int_b)) L[int_b] <- L[int_b] - 1

      log_est <- sum(L * coef_vec)
      var_est <- as.numeric(t(L) %*% vc %*% L)
      se_est  <- sqrt(var_est)
      z_stat  <- log_est / se_est
      p_val   <- 2 * pnorm(-abs(z_stat))

      results[[k]] <- data.frame(
        mut_class    = mc,
        comparison   = paste0(a, "_vs_", b),
        estimate     = exp(log_est),
        log_estimate = log_est,
        std.error    = se_est,
        statistic    = z_stat,
        p.value      = p_val,
        conf.low     = exp(log_est - 1.96 * se_est),
        conf.high    = exp(log_est + 1.96 * se_est),
        stringsAsFactors = FALSE
      )
      k <- k + 1
    }
  }

  # Extra comparisons
  if (!is.null(extra_comparisons)) {
    for (comp in extra_comparisons) {
      a <- comp[1]
      b <- comp[2]

      for (mc in mut_levels) {

        L <- rep(0, length(coef_vec))
        names(L) <- coef_names

        main_a <- get_main_term(a)
        main_b <- get_main_term(b)
        if (!is.na(main_a)) L[main_a] <- L[main_a] + 1
        if (!is.na(main_b)) L[main_b] <- L[main_b] - 1

        int_a <- get_interaction_term(mc, a)
        int_b <- get_interaction_term(mc, b)
        if (!is.na(int_a)) L[int_a] <- L[int_a] + 1
        if (!is.na(int_b)) L[int_b] <- L[int_b] - 1

        log_est <- sum(L * coef_vec)
        var_est <- as.numeric(t(L) %*% vc %*% L)
        se_est  <- sqrt(var_est)
        z_stat  <- log_est / se_est
        p_val   <- 2 * pnorm(-abs(z_stat))

        results[[k]] <- data.frame(
          mut_class    = mc,
          comparison   = paste0(a, "_vs_", b),
          estimate     = exp(log_est),
          log_estimate = log_est,
          std.error    = se_est,
          statistic    = z_stat,
          p.value      = p_val,
          conf.low     = exp(log_est - 1.96 * se_est),
          conf.high    = exp(log_est + 1.96 * se_est),
          stringsAsFactors = FALSE
        )
        k <- k + 1
      }
    }
  }

  results_df        <- bind_rows(results)
  results_df$p_adj  <- p.adjust(results_df$p.value, method = adjust_method)
  results_df$signif <- vapply(results_df$p_adj, sig_label, character(1))

  write.csv(results_df, out_file, row.names = FALSE)
  cat("Wrote:", out_file, "\n")
  return(results_df)
}

# File paths

OPP <- "/shared/team/2025-masters-project/people/justin/all_genomes_opportunity_tables/"
OUT <- "/shared/team/2025-masters-project/people/justin/combined_IRR_tables_with_stats/"

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# 1. Allele frequency
# pairwise = TRUE gives k2_vs_k1 and k3_vs_k1.
# k3_vs_k2 is passed explicitly via extra_comparisons to ensure it is always included regardless of pairwise ordering.

cat("\nRunning allele frequency model...\n")

allele_freq_results <- combined_nb_contrasts(
  file_path         = paste0(OPP, "allele_freq_nb_table.csv"),
  out_file          = paste0(OUT, "allele_freq_combined_IRR_wald.csv"),
  group_var         = "k_class",
  group_levels      = c("k1", "k2", "k3"),
  pairwise          = TRUE,
  extra_comparisons = list(c("k3", "k2"))
)

# 2. Local sequence context

cat("\nRunning sequence context model...\n")

context_results <- combined_nb_contrasts(
  file_path    = paste0(OPP, "context_nb_table.csv"),
  out_file     = paste0(OUT, "context_combined_IRR_wald.csv"),
  group_var    = "context",
  group_levels = c("A_A", "A_C", "A_G", "A_T",
                   "C_A", "C_C", "C_G", "C_T",
                   "G_A", "G_C", "G_G", "G_T",
                   "T_A", "T_C", "T_G", "T_T"),
  pairwise     = FALSE
)

# 3. Genic/intergenic (genic is baseline)

cat("\nRunning genic/intergenic model...\n")

genic_results <- combined_nb_contrasts(
  file_path    = paste0(OPP, "genic_intergenic_nb_table.csv"),
  out_file     = paste0(OUT, "genic_intergenic_combined_IRR_wald.csv"),
  group_var    = "is_igr",
  group_levels = c("FALSE", "TRUE"),
  pairwise     = FALSE
)

# 4. Synonymous / non-synonymous / nonsense (synonymous is baseline)

cat("\nRunning consequence model...\n")

consequence_results <- combined_nb_contrasts(
  file_path    = paste0(OPP, "synonymous_nonsynonymous_nonsense_nb_table.csv"),
  out_file     = paste0(OUT, "consequence_combined_IRR_wald.csv"),
  group_var    = "consequence",
  group_levels = c("synonymous", "non_synonymous", "nonsense"),
  pairwise     = FALSE
)

# 5. Codon position (position 3 is baseline, least constrained)

cat("\nRunning codon position model...\n")

codon_df           <- read.csv(paste0(OPP, "codon_position_nb_table.csv"),
                                stringsAsFactors = FALSE)
codon_df$codon_pos <- as.character(codon_df$codon_pos)

codon_results <- combined_nb_contrasts(
  file_path    = codon_df,
  out_file     = paste0(OUT, "codon_position_combined_IRR_wald.csv"),
  group_var    = "codon_pos",
  group_levels = c("3", "1", "2"),
  pairwise     = FALSE
)

cat("\nAll five models complete. Output files written to:\n", OUT, "\n")
