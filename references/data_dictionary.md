# Data Dictionary

## Raw Features (from source dataset)

| Column | Type | Description | Notes |
|---|---|---|---|
| `longitude` | float | Longitude of block group centroid | More negative = further west |
| `latitude` | float | Latitude of block group centroid | Higher = further north |
| `housing_median_age` | int | Median age of houses in block (years) | Capped at 52 |
| `total_rooms` | int | Total number of rooms in all households | Block-level count |
| `total_bedrooms` | int | Total number of bedrooms in all households | **207 missing values** |
| `population` | int | Total population of block group | |
| `households` | int | Number of households in block group | Denominator for ratio features |
| `median_income` | float | Median household income (tens of thousands of USD) | Range: 0.5–15 |
| `ocean_proximity` | string | Categorical proximity to ocean | 5 categories (see below) |
| `median_house_value` | float | **Target variable** — median home value (USD) | Capped at $500,001 |

### ocean_proximity categories

| Value | Meaning |
|---|---|
| `<1H OCEAN` | Less than 1 hour from the ocean |
| `INLAND` | Inland, not near coast |
| `ISLAND` | On an island |
| `NEAR BAY` | Near San Francisco Bay |
| `NEAR OCEAN` | Near the ocean (>1 hour) |

---

## Engineered Features

| Column | Formula | Rationale |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Per-home room count; raw totals are not comparable across blocks of different sizes |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Fraction of rooms that are bedrooms; lower values indicate more spacious homes |
| `population_per_household` | `population / households` | Average household size; proxy for density and occupancy type |

Raw count columns (`total_rooms`, `total_bedrooms`, `population`, `households`) are
**dropped** after engineering — their information is captured by the ratios.

---

## One-Hot Encoded Features (from ocean_proximity)

After encoding, the following binary columns are added:

- `ocean_proximity_<1H OCEAN`
- `ocean_proximity_INLAND`
- `ocean_proximity_ISLAND`
- `ocean_proximity_NEAR BAY`
- `ocean_proximity_NEAR OCEAN`

`drop_first=False` is used because Random Forest is not affected by
multicollinearity, and keeping all categories improves interpretability
of feature importances.