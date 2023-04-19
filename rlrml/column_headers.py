class Headers:
    def __init__(self, columns):
        #print(columns)
        self.column_headers = {
            'global_headers': {
                'global_begin': 0
            },
            'player_headers': {}
        }
        #print(columns)
        players = set([header.split(' ')[1] for header in columns if header.split(' ')[0] == 'Player'])
        #print(players)
        player = next(header for header in columns if header.split(' ')[1] in players)
        self.column_headers['global_headers']['global_end'] = columns.index(player) - 1
        #global_begin, global_end = self.get_global_headers_as_indices()
        #print(columns[global_begin:global_end+1])
        while len(players) > 0:
            player_begin = next(header for header in columns if header.split(' ')[1] in players)
            player_name = player_begin.split(' ')[1]
            players.remove(player_name)
            #print(players)
            #print(player_name)
            player_header = player_name + '_headers'
            self.column_headers['player_headers'][player_header] = {}
            self.column_headers['player_headers'][player_header]['player_begin'] = columns.index(player_begin)
            if len(players) > 0:
                player_end = next(header for header in columns if header.split(' ')[1] in players)
                self.column_headers['player_headers'][player_header]['player_end'] = columns.index(player_end)-1
            else:
                self.column_headers['player_headers'][player_header]['player_end'] = len(columns)-1

        #print(self.column_headers)
        #print(self.get_player_headers())
        #for indices in self.get_player_headers():
        #    print(columns[indices[0]:indices[1]+1])

        #while player:
        #    print(player)
        #    self.column_headers[player.split(' - ')[0]]['player_begin'] = columns.index(player)
        #    player = None
            #player = next(header for header in columns if header.split(' - ') != player.split(' - '))

    def get_player_headers(self):
        return [
            (self.column_headers['player_headers'][player_dict]['player_begin'],
             self.column_headers['player_headers'][player_dict]['player_end'])
             for player_dict in self.column_headers['player_headers']]

    def get_full_headers(self):
        return self.column_headers

    def get_global_headers(self):
        #print(columns[self.column_headers['global_headers']['global_begin']:self.column_headers['global_headers']['global_end']+1])
        return self.column_headers['global_headers']

    def get_global_headers_as_indices(self):
        return self.column_headers['global_headers']['global_begin'], self.column_headers['global_headers']['global_end']
