import React from 'react';
import { WebSocketContext } from './WebSocketContext';
import { useParams } from 'react-router-dom'

const PlayerDetailPage = () => {
	const { trackerType, trackerId } = useParams()
	const { makeWebsocketRequest } = React.useContext(WebSocketContext);
	const trackerSuffix = `${trackerType}/${trackerId}`
	const [mmr, setMMR] = React.useState(null);

	const handleChange = event => {
		let value = Number(event.target.value)
		setMMR(isNaN(value) ? null : value)
    };

	const handleSetMMRClick = () => {
		makeWebsocketRequest(
			'player_mmr_override', {tracker_suffix: trackerSuffix, mmr}
		)
	};

    return (
        <div>
			{ trackerType }: { trackerId }
			<br />
			<input type="text" value={mmr} onChange={handleChange}
                   placeholder="Enter Desired MMR" />
			<button onClick={handleSetMMRClick}>Set MMR to {mmr}</button>
        </div>
    );
}

export default PlayerDetailPage;
