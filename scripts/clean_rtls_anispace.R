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
  PosObj    = as.data.frame(raw_dt),
  Time.col  = "timestamp",
  ID.col    = "cow_id",
  x.col     = "x",
  y.col     = "y",
  # AniSpace rounds timestamps to whole seconds, so use an integer grid.
  TRes      = 2,
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
  TRes    = 2,
  TCap    = 120,
  tails   = FALSE,
  verbose = TRUE
)

# 5. Extract Tabular Data and Export to Parquet
cat("5. Exporting cleaned AniSpace trajectories...\n")
position_rows <- lapply(seq_along(ani_interp@Pos), function(i) {
  positions <- as.data.frame(ani_interp@Pos[[i]])

  # AniSpace's as.data.frame() method fails when an individual has no
  # positions, so omit empty trajectories while preserving the other rows.
  if (nrow(positions) == 0L) {
    return(NULL)
  }

  # AniSpace does not retain speed in the object. Calculate the step speed
  # from consecutive positions; coordinates are metres and Time is seconds.
  dt <- diff(as.numeric(positions$Time))
  dx <- diff(as.numeric(positions$x))
  dy <- diff(as.numeric(positions$y))
  velocity <- c(NA_real_, sqrt(dx^2 + dy^2) / dt)
  velocity[!is.finite(velocity)] <- NA_real_
  positions$velocity <- velocity

  cbind(
    NIDs = rep(ani_interp@NIDs[i], nrow(positions)),
    IDs = rep(ani_interp@IDs[i], nrow(positions)),
    positions
  )
})
position_rows <- Filter(Negate(is.null), position_rows)

if (length(position_rows) > 0L) {
  clean_dt <- rbindlist(position_rows, use.names = TRUE)
} else {
  clean_dt <- data.table(
    NIDs = integer(),
    IDs = character(),
    Time = integer(),
    x = numeric(),
    y = numeric()
  )
}

# AniSpace stores Time as seconds relative to the first timestamp in TLim.
clean_dt[, datetime := as.POSIXct(
  ani_interp@TLim[1] + Time,
  origin = "1970-01-01",
  tz = "UTC"
)]

output_path <- "data/rtls_cleaned_anispace.parquet"
write_parquet(clean_dt, output_path)

cat("Done! Cleaned spatial data saved to:", output_path, "\n")
