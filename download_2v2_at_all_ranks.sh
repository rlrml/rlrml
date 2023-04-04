#!/usr/bin/env bash
API_KEY="${API_KEY:-ho1llhmFU1qnuKDg5guLAfeyAdSEKR82Nswgtn6M}"
DL_PATH="${DL_PATH:-./data/$(date -Iminutes)}"
DL_COUNT="${DL_COUNT:-5}"
BEFORE_DATE="2023-03-07T00:00:00Z"

echo "Downloading to $DL_PATH"

function download_for_rank {
	rank="$1"
	poetry run \
	   rlbc_download --auth-token "${API_KEY}" \
	   --path "$DL_PATH/$rank" \
	   --count "$DL_COUNT" \
	   -q season f9 -q playlist ranked-doubles \
	   -q replay-date-before "$BEFORE_DATE" \
	   -q min-rank "$rank" -q max-rank "$rank"
}

declare -a Ranks=("grand-champion" "champion" "diamond" "platinum" "gold" "silver")
declare -a Tiers=("1" "2" "3")
for rank_class in "${Ranks[@]}"; do
for tier in "${Tiers[@]}"; do
	download_for_rank "${rank_class}-${tier}"
done
done

download_for_rank "supersonic-legend"
