import React from 'react';
import WebSocketControls from './WebSocketControls';
import { WebSocketContext } from './WebSocketContext';
import GameInfoTable from './GameInfoTable'

const LossAnalysisPage = () => {
    const { makeWebsocketRequest } = React.useContext(WebSocketContext);
    const handleClickStart = () => {
        makeWebsocketRequest('start_loss_analysis')
    }
    return (
        <div>
            <WebSocketControls />
            <button onClick={handleClickStart}>Start Loss Analysis</button>
            <GameInfoTable />
        </div>
    );
};

export default LossAnalysisPage;
