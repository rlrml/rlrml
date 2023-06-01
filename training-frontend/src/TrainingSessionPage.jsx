// TrainingSessionPage.js
import React from 'react';
import WebSocketControls from './WebSocketControls';
import LossChart from './LossChart';

const TrainingSessionPage = () => {
    return (
        <div>
            <WebSocketControls />
            <LossChart />
        </div>
    );
};

export default TrainingSessionPage;
