import Chart from 'chart.js/auto';
import {CategoryScale} from 'chart.js';

import React, { useMemo, useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { useTable, useSortBy } from 'react-table';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

const TrainingSessionPage = () => {
	// For keeping track of connection state.
	const [connectionStatus, setConnectionStatus] = useState('Disconnected');

    // Array to store history of 'loss' values
    const [lossHistory, setLossHistory] = useState([]);

    // Controls the resolution of the graph
    const [bucketSize, setBucketSize] = useState(10);

    // State variable for bucketedLoss
    const [bucketedLossHistory, setBucketedLossHistory] = useState([]);

    // Object to store game info data
    const [gameInfo, setGameInfo] = useState({});

    // WebSocket connection
    const [ws, setWs] = useState(null);

    // WebSocket URL
    const [wsUrl, setWsUrl] = useState('ws://localhost:5002');

        // A function to calculate bucketed average
    const calculateBucketedAverage = (data, bucketSize) => {
        let bucketedData = [];

        for (let i = 0; i < data.length; i += bucketSize) {
            let bucket = data.slice(i, i + bucketSize);
            let sum = bucket.reduce((a, b) => a + b, 0);
            let avg = sum / bucket.length;
            bucketedData.push(avg);
        }

        return bucketedData;
    };

    const connectWebSocket = () => {
        if (wsUrl.trim() !== '') {
            // Initialize WebSocket connection
            const websocket = new WebSocket(wsUrl);
			setConnectionStatus('Connecting...');

			websocket.onopen = () => {
				setConnectionStatus('Connected');
			};

			websocket.onerror = () => {
				setConnectionStatus('Failed to connect');
			};

			websocket.onclose = () => {
				setConnectionStatus('Disconnected');
			};

            // Set WebSocket on this state
            setWs(websocket);
        }
    };

    const sendToggleTraining = () => sendCommand("toggle_training");

    const sendCommand = (commandType) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Sending command " + commandType)
            ws.send(JSON.stringify({ type: commandType }));
        }
    };

    useEffect(() => {
        return () => {
            // Close WebSocket connection when the component unmounts
            if (ws) {
                ws.close();
            }
        };
    }, [ws]);

    useEffect(() => {
        // Attach an event listener for incoming messages
        if (ws) {
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const game_info = Object.fromEntries(data.uuids.map((uuid, index) => [uuid, {
                    "y": data.y[index],
                    "y_pred": data.y_pred[index],
                    "y_loss": data.y_loss[index],
                }]));
                console.log(game_info);
                if (data.loss) {
                    setLossHistory(prevLossHistory => [...prevLossHistory, data.loss]);
                }
                if (game_info) {
                    setGameInfo(prevGameInfo => ({ ...prevGameInfo, ...game_info }));
                }
            };
        }
    }, [ws]);

    useEffect(() => {
        const bucketedData = calculateBucketedAverage(lossHistory, bucketSize);
        setBucketedLossHistory(bucketedData);

    }, [lossHistory, bucketSize]);

    const data = useMemo(() => Object.keys(gameInfo).map(uuid => ({ uuid, ...gameInfo[uuid] })), [gameInfo]);

    const [sortKey, setSortKey] = useState('uuid');
    const [sortDirection, setSortDirection] = useState('asc');

    // Replace with your keys
    const columns = useMemo(() => [
        {
            Header: 'UUID',
            accessor: 'uuid',
        },
        {
            Header: 'Predictions',
            accessor: 'y_pred',
        },
        {
            Header: 'Actual',
            accessor: 'y',
        }
    ], []);

    const {
        getTableProps,
        getTableBodyProps,
        headerGroups,
        rows,
        prepareRow,
    } = useTable({ columns, data }, useSortBy);

    return (
        <div>
			<div>
				<input
					type="text"
					value={wsUrl}
					onChange={(e) => setWsUrl(e.target.value)}
					placeholder="Enter WebSocket URL..."
				/>
				<button onClick={connectWebSocket}>Connect</button>
				<p>Status: {connectionStatus}</p>
                <button onClick={sendToggleTraining}>Toggle Training</button>
                <div style={{ width: '300px', height: '50px' }}>
                    <Slider
                        min={1}
                        max={100} // Adjust this to your needs
                        value={bucketSize}
                        onChange={setBucketSize}
                        style={{ width: '300px', height: '50px' }}
                    />
                </div>
			</div>
            <div style={{ margin: '50px' }}>
            <Line
                data={{
                    labels: bucketedLossHistory.map((_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: bucketedLossHistory,
                            fill: false,
                            backgroundColor: 'rgb(75, 192, 192)',
                            borderColor: 'rgba(75, 192, 192, 0.2)',
                        },
                    ],
                }}
            />
            </div>
            <div>
                <table {...getTableProps()} style={{ margin: '0 auto', marginTop: '50px' }}>
                    <thead>
                        {headerGroups.map(headerGroup => (
                            <tr {...headerGroup.getHeaderGroupProps()}>
                                {headerGroup.headers.map(column => (
                                    <th
                                        {...column.getHeaderProps(column.getSortByToggleProps())}
                                        style={{
                                            borderBottom: 'solid 3px red',
                                            background: 'aliceblue',
                                            color: 'black',
                                            fontWeight: 'bold',
                                        }}
                                    >
                                        {column.render('Header')}
                                        <span>
                                            {column.isSorted
                                                ? column.isSortedDesc
                                                    ? ' 🔽'
                                                    : ' 🔼'
                                                : ''}
                                        </span>
                                    </th>
                                ))}
                            </tr>
                        ))}
                    </thead>
                    <tbody {...getTableBodyProps()}>
                        {rows.map(row => {
                            prepareRow(row);
                            return (
                                    <tr {...row.getRowProps()}>
                                        {row.cells.map(cell => {
                                        return (
                                            <td
                                                {...cell.getCellProps()}
                                                style={{
                                                    padding: '10px',
                                                    border: 'solid 1px gray',
                                                    background: 'papayawhip',
                                                }}
                                            >
                                                {cell.render('Cell')}
                                            </td>
                                        )
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default TrainingSessionPage;
