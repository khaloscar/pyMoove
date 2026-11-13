
# scripts/moo4feed.R
library(arrow)
library(moo4feed)
library(ggplot2)
library(dplyr)

# Set working directory to project root if running from within scripts/
if (basename(getwd()) == "scripts") {
  setwd("..")
}

extdata_path <- "data"    # 🐄 For feed data
fileNames.f <- list.files(
  path       = extdata_path,     # 🐄 Where to look
  pattern    = "^Feed_.*\\.csv$",   # 🐄 Pattern to match for feeder files
  recursive  = TRUE,             # Look in subfolders too
  full.names = TRUE              # Include full path
) |> sort()  

# Configure all global variables in one call
# 👉 EDIT THIS to match your data file structure
set_global_cols(
  # Time zone
  tz = "Europe/Stockholm",
  
  # Column names in your data files
  id_col = "cow",             # animal ID column
  trans_col = "transponder",  # transponder column
  start_col = "start_time",        # visit start time column
  end_col = "end_time",            # visit end time column
  bin_col = "bin",            # bin ID column
  dur_col = "duration",       # duration column
  intake_col = "intake",      # intake column
  start_weight_col = "start_weight",  # start weight column
  end_weight_col = "end_weight",      # end weight column
  
  # Bin settings
  bins_feed = 1:58           # feed bin IDs in your barn
  
  # Physical bin layout for spatial analysis (neighbor detection, synchronicity)
  # Rows are separated by "\n", bins within rows are separated by "-"
  # Only bins in the same row are spatial neighbors (left/right)
)


col_names <- c("cow","bin","start_time","intake","duration","transponder","end_time", "start_weight", "end_weight")
select_cols <- c("cow","bin","start_time","intake","duration","transponder","end_time", "start_weight", "end_weight")

# Process all feeder files
all_fed <- process_all_feed(
    files = fileNames.f,
    col_names   = col_names,     # Column names for your data
    # drop_ids    = drop_ids,      # IDs to remove
    # select_cols = select_cols,   # Columns to keep (from above)
    sep         = ",",           # 👉 EDIT if your files use a different separator (tab="\t")
    header      = TRUE,         # 👉 Change to TRUE if your files have header rows
    adjust_dst  = TRUE           # Set to FALSE if you don't want DST corrections
)

my_qc_config <- qc_config(
  # Feed thresholds
  high_dur_feed = 2000,          # Flag feed visits longer than 2000 seconds
  large_intake_visit_feed = 8,   # Flag feed visits with more than 8kg intake
  large_intake_rate_feed = 0.008, # Flag feed intake rates above 0.008 kg/s
  low_feed_intake = 35,          # Flag cows eating less than 35kg/day
  high_feed_intake = 75,         # Flag cows eating more than 75kg/day
  
  # General settings
  low_visit_threshold = 10,      # Flag bins with fewer than 10 visits/day
  total_cows_expected = 461,      # Set to your herd size
  replacement_threshold = 26,    # Time gap (s) to classify replacement behavior
  calibration_error = 0.5        # Feeder/drinkercalibration error (kg/L)
)

# Run comprehensive quality check
qc_results <- qc(
  feed = all_fed,               # Your processed feeder data
  cfg = my_qc_config,           # Your custom thresholds
  verbose = FALSE,               # Do not show detailed warning messages about errorous visits, set to TRUE to see them
  fix_double_detections = TRUE  # Automatically fix overlapping visits
)

# Store warnings for review
warning <- qc_results$warnings
qc_feed <- qc_results$feed

# Apply KNN outlier detection
feed_with_outliers <- knn_clean_feed(
  qc_feed,            # Our cleaned feed data
  k = 50,                # Number of neighbors to consider
  threshold_percentile = 99.3,  # Threshold for outlier detection
  # Custom scaling helps balance the influence of each variable
  custom_scaling = list(
    rate = 10000,        # Scaling for intake rate
    intake = 7,          # Scaling for intake amount
    duration = 0.03      # Scaling for duration
  ),
  remove_outliers = FALSE # Set to TRUE if you wish to remove outliers
)

# Visualize KNN outlier detection
p_knn <- viz_outliers(
  feed_with_outliers,
  x_var = "rate",
  y_var = "intake",
  title = "Feed Outliers: Rate vs Intake",
  jitter_amount = 0,
  regular_color = "lightgreen",
  outlier_color = "orange"
)

# print(p_knn)

# Apply KNN outlier detection
clean_feed <- knn_clean_feed(
  qc_feed,            # Our cleaned feed data
  k = 50,                # Number of neighbors to consider
  threshold_percentile = 99.3,  # Threshold for outlier detection
  # Custom scaling helps balance the influence of each variable
  custom_scaling = list(
    rate = 10000,        # Scaling for intake rate
    intake = 7,          # Scaling for intake amount
    duration = 0.03      # Scaling for duration
  ),
  remove_outliers = TRUE # Set to TRUE if you wish to remove outliers
)

# Process replacement events for all days
replacements <- record_replacement_days(
  comb = clean_feed,           # Our cleaned feed data
  cfg = qc_config(replacement_threshold = 26)      # Time gap (seconds) to classify replacement behavior
)

# Examine the first few replacement events
head(replacements[[1]])

# Summary of replacement events
cat("Replacement events per day:\n")
sapply(replacements, nrow)

# Export replacement data to Parquet and CSV per day
cat("Exporting replacement events to Parquet and CSV...\n")

# Ensure replacements is a list by day
replacement_list <- if (is.data.frame(replacements)) list(replacements) else replacements

for (day_name in names(replacement_list)) {
  day_df <- replacement_list[[day_name]]
  
  # Determine date string for the filename
  raw_date <- if (!is.null(day_df$date) && length(day_df$date) > 0) {
    as.character(day_df$date[1])
  } else if (!is.null(day_df$Date) && length(day_df$Date) > 0) {
    as.character(day_df$Date[1])
  } else {
    day_name
  }
  date_str <- gsub("-", "", raw_date)
  
  # Remove the "Date" column
  day_df <- day_df |> select(-any_of(c("date", "Date")))
  
  # Convert duration to numeric seconds for clean parquet and csv serialization
  if ("bout_interval" %in% names(day_df)) {
    day_df$bout_interval <- as.numeric(day_df$bout_interval)
  }
  
  parquet_path <- sprintf("data/feed_replacements_%s.parquet", date_str)
  csv_path     <- sprintf("data/feed_replacements_%s.csv", date_str)
  
  write_parquet(day_df, parquet_path)
  write.csv(day_df, csv_path, row.names = FALSE)
  
  cat("Done! Replacement data saved to:\n  -", parquet_path, "\n  -", csv_path, "\n")
} 

