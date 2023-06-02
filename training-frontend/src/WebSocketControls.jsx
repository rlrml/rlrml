// WebSocketControls.js
import React from 'react';
import { WebSocketContext } from './WebSocketContext';

const WebSocketControls = () => {
    const { setWebSocketAddress, connectionStatus, webSocket, makeWebsocketRequest } = React.useContext(WebSocketContext);

	const [address, setAddress ] = React.useState('ws://localhost:5002');

    const handleClickStart = () => {
        makeWebsocketRequest('start_training')
    };

    const handleClickStop = () => {
        makeWebsocketRequest('stop_training')
    };

    const handleClickSave = () => {
        makeWebsocketRequest('save_model')
    }

    const handleChange = event => {
		setAddress(event.target.value)
    };

    const handleConnect = () => {
        setWebSocketAddress(address);
    };

    return (
        <div>
            <input type="text" value={address} onChange={handleChange}
                   placeholder="Enter WebSocket address" />
            <button onClick={handleConnect}>Connect ({connectionStatus})</button>
            <button onClick={handleClickStart}>Start Training</button>
            <button onClick={handleClickStop}>Stop Training</button>
            <button onClick={handleClickSave}>Save Model</button>
        </div>
    );
};

export default WebSocketControls;
