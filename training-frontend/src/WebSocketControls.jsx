// WebSocketControls.js
import React from 'react';
import { WebSocketContext } from './WebSocketContext';

const WebSocketControls = () => {
    const { setWebSocketAddress, connectionStatus, webSocket } = React.useContext(WebSocketContext);

	const [address, setAddress ] = React.useState('ws://localhost:5002');

    const handleClickToggle = () => {
        console.log(webSocket);
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            webSocket.send(JSON.stringify({ type: 'toggle_training' }));
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
            <button onClick={handleClickToggle}>Toggle Training</button>
        </div>
    );
};

export default WebSocketControls;
