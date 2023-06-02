import React from 'react';
import { WebSocketContext } from './WebSocketContext';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from '@tanstack/react-table';
import { Link } from "react-router-dom";
import _ from 'lodash';

function getPlayerCellContents(player) {
	const actual = Math.trunc(Number(player.mmr)).toString().padStart(4, '0');
	const predicted = Math.trunc(Number(player.prediction)).toString().padStart(4, '0');
	const playerText = player.tracker_suffix.split('/')[1].substring(0, 10).padStart(10, ' ');
	const trackerLinkTarget = `https://rocketleague.tracker.network/rocket-league/profile/${player.tracker_suffix}`;
	const predictedColor = player.mask == 0 ? "grey" : "black";
	return (
		<span>
			<Link to={`/player_detail/${player.tracker_suffix}`}>{playerText}
			</Link> - <a href={trackerLinkTarget}>{actual}</a> <span style={{ color: predictedColor }}>({predicted})</span>
		</span>
	);
}

function getLargestMiss(row) {
	let pairs = _.zip(row.y, row.y_pred, row.masks);
	let deltas = pairs.map(
		v => v[2] === 0 ? 0 : Math.abs(v[0] - v[1]),
	)
	return Math.trunc(Math.max(...deltas))
}

function getLargestDelta(row) {
	let excludingMasked = _.zip(row.y, row.masks).filter((v) => v[1] != 0).map((v) => v[0]);
	let max = Math.max(...excludingMasked);
	let min = Math.min(...excludingMasked);
	return Math.trunc(max - min);
}

function meanSquaredError(y_true, y_pred, mask) {
	let sum = 0;
	let length = mask.reduce((a, b) => a + b, 0);

	for (let i = 0; i < length; i++) {
		if (mask[i] === 0) {
			continue
		}
		let diff = y_true[i] - y_pred[i];
		sum += diff * diff;
	}

	return sum / length;
}

function meanAbsoluteError(y_true, y_pred, mask) {
	let sum = 0;
	let length = mask.reduce((a, b) => a + b, 0);

	for (let i = 0; i < length; i++) {
		if (mask[i] === 0) {
			continue
		}
		let diff = Math.abs(y_true[i] - y_pred[i]);
		sum += diff;
	}

	return sum / length;
}

function ballchasingURL(uuid) {
	return `https://ballchasing.com/replay/${uuid}`
}

const GameInfoTable = () => {
	const { gameInfo, trainingPlayerCount, sorting, setSorting } = React.useContext(WebSocketContext);

	const columns = React.useMemo(
		() => [
			{
				header: 'UUID',
				accessorFn: row => (
					<span>
						<Link to={`game_detail/${row.uuid}`}>{row.uuid.substring(0, 8)}</Link>
						<a href={ballchasingURL(row.uuid)}>(bc)</a>
					</span>
				),
				cell: row => row.getValue(),
			},
			{
				header: 'Players',
				columns: [...Array(trainingPlayerCount).keys()].map((playerIndex) => {
					return {
						header: playerIndex.toString(),
						accessorFn: (row) => {
							return getPlayerCellContents(row.players[playerIndex])
						},
						cell: row => row.getValue(),
					}
				}),
			},
			{
				header: 'Upd. Ep.',
				accessorKey: 'update_epoch',
			},
			{
				header: '> Miss',
				accessorFn: getLargestMiss,
			},
			{
				header: '> Delt',
				accessorFn: getLargestDelta,
			},
			{
				header: 'MSE',
				accessorFn: row => Math.trunc(meanSquaredError(row.y_pred, row.y, row.masks)),
			},
			{
				header: 'MAE',
				accessorFn: row => Math.trunc(meanAbsoluteError(row.y_pred, row.y, row.masks)),
			},
			{
				header: 'Mask',
				accessorFn: row => row.masks.reduce((a, b) => a + b, 0),
			},
		],
		[trainingPlayerCount]
	);

	const data = React.useMemo(() => Object.keys(gameInfo).map(uuid => ({ uuid, ...gameInfo[uuid] })), [gameInfo]);

	const table = useReactTable({
		data,
		columns,
		state: {
			sorting,
		},
		onSortingChange: setSorting,
		getCoreRowModel: getCoreRowModel(),
		getSortedRowModel: getSortedRowModel(),
	});

   return (
    <div>
      <div />
      <table>
        <thead>
          {table.getHeaderGroups().map(headerGroup => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map(header => {
                return (
                  <th key={header.id} colSpan={header.colSpan}>
                    {header.isPlaceholder ? null : (
                      <div
                        {...{
                          className: header.column.getCanSort()
                            ? 'cursor-pointer select-none'
                            : '',
                          onClick: header.column.getToggleSortingHandler(),
                        }}
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {{
                          asc: ' 🔼',
                          desc: ' 🔽',
                        }[header.column.getIsSorted()] ?? null}
                      </div>
                    )}
                  </th>
                )
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table
            .getRowModel()
            .rows.slice(0, 100)
            .map(row => {
              return (
                <tr key={row.id}>
                  {row.getVisibleCells().map(cell => {
                    return (
						<td key={cell.id} style={{textAlign: "center"}}>
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
        </tbody>
      </table>
      <div>{table.getRowModel().rows.length} Rows</div>
      <pre>{JSON.stringify(sorting, null, 2)}</pre>
    </div>
  );
};

export default GameInfoTable;
