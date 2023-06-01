// TrainingSessionPage.js
import React from 'react';
import WebSocketControls from './WebSocketControls';
import LossChart from './LossChart';
import GameInfoTable from './GameInfoTable'

const TrainingSessionPage = () => {
    return (
        <div>
            <WebSocketControls />
            <LossChart />
            <GameInfoTable />
        </div>
    );
};

export default TrainingSessionPage;
