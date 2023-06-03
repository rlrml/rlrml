import React from 'react';
import { WebSocketContext } from './WebSocketContext';
import { useParams } from 'react-router-dom'

import { getPlayerHTML } from './Util';

const GameDetailPage = () => {
	const { uuid } = useParams()
	const { makeWebsocketRequest, gameInfo } = React.useContext(WebSocketContext)
	const [reason, setReason] = React.useState(null)
	const game = gameInfo[uuid];

	const handleReasonChange = event => {
		setReason(event.target.value)
    };

	const handleBlacklistReplayClick = () => {
		makeWebsocketRequest('blacklist_replay', { uuid, reason })
	}

    return (
        <div>
			{ uuid }
			<br />
			<button onClick={handleBlacklistReplayClick}>Blacklist Replay for {reason}</button>
			<select value={reason} onChange={handleReasonChange}>
				<option value="throwing">Throwing</option>
				<option value="afk">AFK</option>
				<option value="disconnect">Disconnect/Leave</option>
				<option value="game length">Game Length</option>
				<option value="other">Other</option>
			</select>
			<br />
			<br />
			<pre>
				{ game.players.map(getPlayerHTML) }
				<br />
				{ JSON.stringify(gameInfo[uuid], null, 3) }
			</pre>
        </div>
    );
}

export default GameDetailPage;
