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

function formatMMRArrayWithIndex(array, separator = " | ") {
	const formattedValues = array.map((value, index) => {
		const truncatedValue = Math.trunc(value);
		const padded = truncatedValue.toString().padStart(4, '0');
		return `${index + 1}: ${padded}`;
  });

  return formattedValues.join(separator);
}

function getLargestMiss(row) {
	let pairs = _.zip(row.y, row.y_pred);
	let deltas = pairs.map(
		v => Math.abs(v[0] - v[1]),
	)
	return Math.max(...deltas)
}

function getLargestDelta(row) {
	let max = Math.max(...row.y);
	let min = Math.min(...row.y);
	return max - min;
}

const zip = (a, b) => a.map((k, i) => [k, b[i]]);

const GameInfoTable = () => {
	const [sorting, setSorting] = React.useState([]);

	const { gameInfo } = React.useContext(WebSocketContext);

	const columns = React.useMemo(
		() => [
			{
				header: 'UUID',
				accessorKey: 'uuid',
			},
			{
				header: 'Actual MMRs',
				accessorFn: row => formatMMRArrayWithIndex(row.y),
			},
			{
				header: 'Predicted MMRs',
				accessorFn: row => formatMMRArrayWithIndex(row.y_pred),
			},
			{
				header: 'Updated Epoch',
				accessorKey: 'update_epoch',
			},
			{
				header: 'Largest Miss',
				accessorFn: getLargestMiss,
			},
			{
				header: 'Largest Delta',
				accessorFn: getLargestDelta,
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
		debugTable: true,
	});

   return (
    <div className="p-2">
      <div className="h-2" />
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
                      <td key={cell.id}>
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
