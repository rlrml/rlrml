// WebSocketControls.js
import React from 'react';
import { WebSocketContext } from './WebSocketContext';

const WebSocketControls = () => {
    const { setWebSocketAddress, connectionStatus, webSocket } = React.useContext(WebSocketContext);

	const [address, setAddress ] = React.useState('ws://localhost:5002');

    const handleClickStart = () => {
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            webSocket.send(JSON.stringify({ type: 'start_training' }));
        }
    };

    const handleClickStop = () => {
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            webSocket.send(JSON.stringify({ type: 'stop_training' }));
        }
    };

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
        </div>
    );
};

export default WebSocketControls;
