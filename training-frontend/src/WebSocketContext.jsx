import React from 'react';
import _ from 'lodash';

const WebSocketContext = React.createContext(null);

const WebSocketProvider = ({ children }) => {
  const [connectionStatus, setConnectionStatus] = React.useState('disconnected');
  const [webSocketAddress, setWebSocketAddress] = React.useState(null);
  const [webSocket, setWebSocket] = React.useState(null);

  const [lossHistory, setLossHistory] = React.useState([]);
  const [gameInfo, setGameInfo] = React.useState({});
  const [trainingPlayerCount, setTrainingPlayerCount] = React.useState(4);

  const [configuration, setConfiguration] = React.useState({});
  let socket;

  const handleTrainingEpoch = (data) => {
    setLossHistory(prevLossHistory => [...prevLossHistory, data.loss]);
    setGameInfo(prevGameInfo => ({...prevGameInfo, ...getGameInfo(data)}));
  };

  const handleTrainingStart = (data) => {
    if (isNaN(data.player_count)) {
      console.log(`Data did not contain numerical player count ${data}`)
    }
    setTrainingPlayerCount(data.player_count);
  }

  const messageTypeToHandler = {
    "training_epoch": handleTrainingEpoch,
    "training_start": handleTrainingStart,
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
	<WebSocketContext.Provider value={{ lossHistory, gameInfo, connectionStatus, setWebSocketAddress, webSocket, trainingPlayerCount }}>
      {children}
    </WebSocketContext.Provider>
  );
};

function getGameInfo(data) {
  return Object.fromEntries(data.uuids.map((uuid, index) => [uuid, {
    "uuid": uuid,
    "players": _.zipWith(
      data.tracker_suffixes[index], data.y[index], data.y_pred[index],
      (tracker_suffix, mmr, prediction) => {
        return {tracker_suffix, mmr, prediction};
      }
    ),
    "y": data.y[index],
    "y_pred": data.y_pred[index],
    "y_loss": data.y_loss[index],
    "tracker_suffixes": data.tracker_suffixes[index],
    "update_epoch": data.epoch,
  }]));
}

export { WebSocketContext, WebSocketProvider };
