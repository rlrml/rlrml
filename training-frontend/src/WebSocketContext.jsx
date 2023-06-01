import React from 'react';
import _ from 'lodash';

const WebSocketContext = React.createContext(null);

const WebSocketProvider = ({ children }) => {
  const [connectionStatus, setConnectionStatus] = React.useState('disconnected');
  const [webSocketAddress, setWebSocketAddress] = React.useState(null);
  const [webSocket, setWebSocket] = React.useState(null);

  const [lossHistory, setLossHistory] = React.useState([]);
  const [gameInfo, setGameInfo] = React.useState({});

  const [configuration, setConfiguration] = React.useState({});
  let socket;

  const handleTrainingEpoch = (data) => {
    setLossHistory(prevLossHistory => [...prevLossHistory, data.loss]);
    setGameInfo(prevGameInfo => ({...prevGameInfo, ...getGameInfo(data)}));
  };

  const messageTypeToHandler = {
    "training_epoch": handleTrainingEpoch,
  };

  React.useEffect(() => {
    if (!webSocketAddress) {
      return;
    }

    socket = new WebSocket(webSocketAddress)
    setWebSocket(socket);

    socket.onopen = () => {
      setConnectionStatus('connected');
    };

    socket.onerror = () => {
      setConnectionStatus('error');
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type in messageTypeToHandler) {
        messageTypeToHandler[message.type](message.data);
      } else {
        console.log(`Unable to handle message of type ${message.type}`)
      }
    };

	socket.onclose = () => {
	  setConnectionStatus('disconnected');
      setWebSocketAddress(null);
	};

    return () => {
      socket.close();
      setWebSocketAddress(null);
    };

  }, [webSocketAddress]);

  return (
	<WebSocketContext.Provider value={{ lossHistory, gameInfo, connectionStatus, setWebSocketAddress, webSocket }}>
      {children}
    </WebSocketContext.Provider>
  );
};

function getGameInfo(data) {
  return Object.fromEntries(data.uuids.map((uuid, index) => [uuid, {
    "uuid": uuid,
    "players": _.zipObject(
      ["tracker_suffix", "mmr", "prediction"],
      [data.tracker_suffixes[index], data.y[index], data.y_pred[index]]
    ),
    "y": data.y[index],
    "y_pred": data.y_pred[index],
    "y_loss": data.y_loss[index],
    "tracker_suffixes": data.tracker_suffixes[index],
    "update_epoch": data.epoch,
  }]));
}

export { WebSocketContext, WebSocketProvider };
