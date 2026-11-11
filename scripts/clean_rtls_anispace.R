# scripts/clean_rtls_anispace.R

library(arrow)
library(data.table)
library(AniSpace)

# Set working directory to project root if running from within scripts/
if (basename(getwd()) == "scripts") {
  setwd("..")
}

cat("1. Loading raw RTLS dataset from Parquet...\n")
raw_dt <- as.data.table(read_parquet("data/rtls_raw_combined.parquet"))
cat("Loaded", nrow(raw_dt), "rows for", uniqueN(raw_dt$cow_id), "cows.\n")

# 2. Transform into AniSpace Object via load.Space()
cat("2. Constructing AniSpace spatial object...\n")
ani_obj <- load.Space(
  PosObj    = raw_dt,
  Time.col  = "timestamp",
  ID.col    = "cow_id",
  x.col     = "x",
  y.col     = "y",
  TRes      = 2.22,
  Temp.sort = TRUE,
  verbose   = TRUE
)

# 3. Speed-based Spike Filtering via spikes()
cat("3. Filtering trajectory spikes (> 2.5 m/s)...\n")
ani_clean <- spikes(
  AniObj    = ani_obj,
  method    = "filter",
  max.speed = 2.5,
  verbose   = TRUE
)

# 4. Gap Interpolation via interpolate()
cat("4. Interpolating missing fixes (Makima algorithm)...\n")
ani_interp <- interpolate(
  AniObj  = ani_clean,
  method  = "makima",
  TRes    = 2.22,
  TCap    = 120,
  tails   = FALSE,
  verbose = TRUE
)

# 5. Extract Tabular Data and Export to Parquet
cat("5. Exporting cleaned AniSpace trajectories...\n")
clean_df <- as.data.frame(ani_interp)
clean_dt <- as.data.table(clean_df)

clean_dt[, datetime := as.POSIXct(Time, origin = "1970-01-01", tz = "UTC")]

output_path <- "data/rtls_cleaned_anispace.parquet"
write_parquet(clean_dt, output_path)

cat("Done! Cleaned spatial data saved to:", output_path, "\n")
