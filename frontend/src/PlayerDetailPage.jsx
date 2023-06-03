import React from 'react';
import { WebSocketContext, GameInfoContext } from './WebSocketContext';
import GameInfoTable from './GameInfoTable'
import { useParams } from 'react-router-dom'
import _ from 'lodash';

const PlayerDetailPage = () => {
	const { trackerType, trackerId } = useParams()
	const { makeWebsocketRequest, gameInfo } = React.useContext(WebSocketContext);
	const trackerSuffix = `${trackerType}/${trackerId}`
	const [mmr, setMMR] = React.useState(null);

	const relevantGames = Object.fromEntries(
		_.filter(gameInfo, (game) => {
			let matching = _.find(game.players, (player) =>
				player.tracker_suffix == trackerSuffix
			)
			return matching !== undefined
		}
	).map((game) => [game.uuid, game]));

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
			<GameInfoContext.Provider value={{ gameInfo: relevantGames }}>
				<GameInfoTable />
			</GameInfoContext.Provider>
			<input type="text" value={mmr} onChange={handleChange}
                   placeholder="Enter Desired MMR" />
			<button onClick={handleSetMMRClick}>Set MMR to {mmr}</button>
        </div>
    );
}

export default PlayerDetailPage;
