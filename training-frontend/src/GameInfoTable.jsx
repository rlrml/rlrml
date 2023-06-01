import React from 'react';
import { WebSocketContext } from './WebSocketContext';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from '@tanstack/react-table'
import _ from 'lodash';

function formattedPlayerToPrediction(row) {
	let playerData = _.zip(...[row.tracker_suffixes, row.y, row.y_pred]);
	const formattedValues = playerData.map((value) => {
		const actual = Math.trunc(Number(value[1])).toString().padStart(4, '0');
		const predicted = Math.trunc(Number(value[2])).toString().padStart(4, '0');
		const playerText = value[0].split('/')[1].substring(0, 10).padStart(10, ' ');
		const linkTarget = `https://rocketleague.tracker.network/rocket-league/profile/${value[0]}`;
		return (
			<span>
				<a href={linkTarget}>{playerText}</a> - {actual} ({predicted})
			</span>
		);
	});

	return (
		<span>
			{ formattedValues }
		</span>
	)
}

function getLargestMiss(row) {
	let pairs = zip(row.y, row.y_pred);
	let deltas = pairs.map(
		v => Math.abs(v[0] - v[1]),
	)
	return Math.trunc(Math.max(...deltas))
}

function getLargestDelta(row) {
	let max = Math.max(...row.y);
	let min = Math.min(...row.y);
	return Math.trunc(max - min);
}

function meanSquaredError(y_true, y_pred) {
  let sum = 0;
  let length = y_true.length;

  // Check if lengths of arrays are the same
  if (length !== y_pred.length) {
    throw new Error("Arrays should have the same length.");
  }

  for (let i = 0; i < length; i++) {
    let diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }

  return sum / length;
}

function meanAbsoluteError(y_true, y_pred) {
  let sum = 0;
  let length = y_true.length;

  // Check if lengths of arrays are the same
  if (length !== y_pred.length) {
    throw new Error("Arrays should have the same length.");
  }

  for (let i = 0; i < length; i++) {
    let diff = Math.abs(y_true[i] - y_pred[i]);
    sum += diff;
  }

  return sum / length;
}

const zip = (a, b) => a.map((k, i) => [k, b[i]]);

const GameInfoTable = () => {
	const [sorting, setSorting] = React.useState([]);

	const { gameInfo } = React.useContext(WebSocketContext);

	const columns = React.useMemo(
		() => [
			{
				header: 'UUID',
				accessorFn: row => row.uuid.substring(0, 8) ,
			},
			{
				header: 'Players',
				accessorFn: formattedPlayerToPrediction,
				cell: row => row.getValue(),
			},
			{
				header: 'Players2',
				accessorFn: formattedPlayerToPrediction,
				cell: row => console.log(row),
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
				accessorFn: row => Math.trunc(meanSquaredError(row.y_pred, row.y)),
			},
			{
				header: 'MAE',
				accessorFn: row => Math.trunc(meanAbsoluteError(row.y_pred, row.y)),
			}
		],
		[]
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
