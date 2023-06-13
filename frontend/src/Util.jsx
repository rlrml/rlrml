import { Link } from "react-router-dom";

function trackerUrl(trackerSuffix) {
    return `https://rocketleague.tracker.network/rocket-league/profile/${trackerSuffix}`
}

function getPlayerHTML(player) {
	let actual = Math.trunc(
        Number(player.mmr)
    ).toString().padStart(4, '0');
	const predicted = Math.trunc(
        Number(player.prediction)
    ).toString().padStart(4, '0');
	const playerText = player.tracker_suffix.split('/')[1].substring(0, 10);
	const trackerLinkTarget = trackerUrl(player.tracker_suffix);

    let actualColor = "blue";
	let predictedColor = "black";

	if (player.mask === 0) {
		actual = "N/A";
		actualColor = "grey";
	}

	if (player.isBiggestMiss) {
		predictedColor = "red";
	}

	return (
		<span>
			<Link to={`/player_detail/${player.tracker_suffix}`}>{playerText}
			</Link> - <a href={trackerLinkTarget} style={{ color: actualColor }}>{actual}
					  </a> <span style={{ color: predictedColor }}>({predicted})</span>
		</span>
	);
}

function meanSquaredError(y_true, y_pred, mask) {
	let sum = 0;
	let length = mask.reduce((a, b) => a + b, 0);

	for (let i = 0; i < length; i++) {
		if (mask[i] === 0) {
			continue
		}
		let diff = y_true[i] - y_pred[i];
		sum += diff * diff;
	}

	return sum / length;
}

function meanAbsoluteError(y_true, y_pred, mask) {
	let sum = 0;
	let length = mask.reduce((a, b) => a + b, 0);

	for (let i = 0; i < length; i++) {
		if (mask[i] === 0) {
			continue
		}
		let diff = Math.abs(y_true[i] - y_pred[i]);
		sum += diff;
	}

	return sum / length;
}

export { getPlayerHTML, trackerUrl, meanSquaredError, meanAbsoluteError };
