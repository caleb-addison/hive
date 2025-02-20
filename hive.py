import numpy as np
import copy
from typing import List, Optional, Tuple, Set

# Use axial coordinates: each position is represented as (q, r)
Coordinate = Tuple[int, int]

class Tile:
    def __init__(self, color: str, tile_type: str, tile_id: int = 1, axial: Optional[Coordinate] = None, height: int = 0, covered: bool = False):
        """
        Base class for a Hive tile.

        :param color: String representing the tile's color (e.g., 'white' or 'black')
        :param tile_type: A string representing the type of tile (e.g., 'Queen Bee', 'Beetle', etc.)
        :param tile_id: An integer indiciating which instance of a tile this is. Used to track distinct tiles of the same 'type'. Indexed to 1.
        :param axial: Axial coordinate (q, r) of the tile; if None, the tile is not yet placed.
        :param height: Vertical height (0 means on the board's base, >0 means stacked on top of another tile)
        """
        self.color = color
        self.tile_type = tile_type
        self.tile_id = tile_id  # Deafults to 1
        self.axial = axial  # None if not placed
        self.height = height  # Board height (for stacking)
        self.covered = covered  # Is this piece covered by other, stored so we don't need to recompute with every move.
        self.valid_moves: Set[Coordinate] = []  # This set should be updated based on the game state

    def move(self, new_axial: Coordinate) -> None:
        """
        Moves the tile to a new axial coordinate.
        (Additional move validations should be done in the game state logic.)
        """
        self.axial = new_axial
        # In a more complex version you may need to update 'height' if the move involves stacking.
        # TODO

    def update_valid_moves(self, game_state: "HiveGameState") -> None:
        """
        Update the list of valid moves for this tile based on the current game state.
        """
        
        if self.axial is None:
            if game_state.players[game_state.current_player_index] != self.color:
                self.valid_moves = set()  # Unplaced piece that doesn't belong to current player has no valid moves
            elif game_state.turn in [0, 1] and self.tile_type == "Queen":
                self.valid_moves = set()  # Queen cannot be placed on either player's first turn
            elif (
                game_state.turn in [6, 7]
                and not game_state.queen_placed[game_state.current_player_index]
                and self.tile_type != "Queen"
            ):
                self.valid_moves = set() # By each player's 4th turn they must have played their bee. Force this placement on turn 4 if necessary.
            else:
                self.valid_moves = game_state.get_placements(self.color)  # Unplaced piece belonging to current player, can be placed

        elif game_state.queen_placed[game_state.current_player_index]:
            self.valid_moves = game_state.get_moves(self)  # Can only move pieces once your bee is placed
        else:
            self.valid_moves = set()

        # print(f'updating moves for {self} : {self.valid_moves}')

    def __repr__(self) -> str:
        h = 'h' + str(self.height) + '(' + ('X' if self.covered else ' ') + ')'
        return f"{self.color} {self.tile_type} at {self.axial} {h}"


class HiveGameState:
    def __init__(self):
        """
        Represents the overall state of a Hive game.

        Attributes:
        - tiles: a list of all Tile objects (both placed on the board and not yet placed)
        - last_move: the reference of the most recently moved piece.
        - history: a list of strings where history[i] shows the state of the game on turn [i]
        - players: a simple list of player colors (you can later substitute a full Player class if desired)
        - current_player_index: index in the players list indicating whose turn it is.
        - turn: tracks the number of turns in the game. Turn = 0 indicates the game has not started. Turn = 1 indicated one turn has been taken, and so on.
        - outcome: a string indicating the outcome of the game (W = white win, B = black win, DR = draw to repetition, DQ = draw to both queens being simultaneously surrounded).
        """
        self.tiles: List[Tile] = []
        self.last_move: Optional[Tile] = None
        self.history: List[str] = []  # The game state is flattened into a string each turn
        self.players = ['white', 'black']
        self.queen_placed = [False, False]  # Track if player at that index has placed their queen
        self.current_player_index = 0
        self.turn = 0
        self.outcome: str = None

    @property
    def current_player(self) -> str:
        return self.players[self.current_player_index]

    def set_current_player(self, player) -> None:
        self.current_player_index = self.players.index(player)

    def add_tile(self, tile: Tile) -> None:
        """
        Adds a tile to the game state.

        Before adding, this method checks that there is no other tile with the same
        color, tile_type, and tile_id already present in the game.
        """
        # Check for duplicates using the unique key (color, tile_type, tile_id)
        for existing_tile in self.tiles:
            if (
                    existing_tile.color == tile.color and
                existing_tile.tile_type == tile.tile_type and
                existing_tile.tile_id == tile.tile_id
            ):
                raise ValueError(
                    f"Tile with color '{tile.color}', type '{tile.tile_type}', and id {tile.tile_id} already exists."
                )
        self.tiles.append(tile)

    def get_tile_by_id(self, color: str, tile_type: str, tile_id: int) -> Optional[Tile]:
        """
        Retrieve a specific tile from the game state using its unique identifier.

        :param color: The color of the tile (e.g., 'white' or 'black')
        :param tile_type: The type of the tile (e.g., 'Queen Bee', 'Beetle', etc.)
        :param tile_id: The instance identifier for this tile.
        :return: The Tile object if found, or None if no matching tile exists.
        """
        for tile in self.tiles:
            if tile.color == color and tile.tile_type == tile_type and tile.tile_id == tile_id:
                return tile
        return None

    def get_tile_at(self, axial: Coordinate) -> Optional[Tile]:
        """
        Returns the top-most tile (the one with the highest height) at a given axial coordinate.
        If no tile is at that position, returns None.
        """
        if axial is None:
            return None
            
        tiles_here = [tile for tile in self.tiles if tile.axial == axial]
        if not tiles_here:
            return None
        # Return the tile with the highest height (top of the stack)
        return max(tiles_here, key=lambda t: t.height)

    def get_all_tiles_at(self, axial: Coordinate) -> List[Tile]:
        """
        Returns all tiles (in all heights) at a given axial coordinate.
        """
        return [tile for tile in self.tiles if tile.axial == axial]

    def get_adjacent_spaces(self, axial: Coordinate) -> List[Tuple[Coordinate, Optional[Tile]]]:
        """
        Given an axial coordinate, returns a list of adjacent coordinates along with the top-most tile at each.

        Each entry in the returned list is a tuple:
        - First element: The coordinate of the adjacent hex.
        - Second element: The top-most tile at that coordinate, or None if the space is empty.

        :param axial: The axial coordinate (q, r) of the reference tile.
        :return: A list of tuples (Coordinate, Optional[Tile]).
        """
        if axial is None:
            return []

        q, r = axial
        # The six hex directions in axial coordinates
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        adjacent_spaces = []
        for dq, dr in directions:
            neighbor = (q + dq, r + dr)
            tile = self.get_tile_at(neighbor)  # Get the top-most tile at this coordinate (if any)
            adjacent_spaces.append((neighbor, tile))

        return adjacent_spaces

    def move_tile(self, tile: Tile, new_axial: Coordinate) -> bool:
        """
        Attempts to move a tile to a new axial coordinate.
        Checks if the new coordinate is in the tile's list of valid moves.

        Returns True if the move is performed, False otherwise.
        """
        # print(f'move_tile called: {tile} -> {new_axial}')
        if new_axial in tile.valid_moves:
            # Store info relating to the old/new coordinates
            old_pos = tile.axial
            covered_tile = self.get_tile_at(new_axial)
            # Move the tile
            tile.move(new_axial)
            tile.height = covered_tile.height + 1 if covered_tile is not None else 0
            self.last_move = tile
            # Track if the queen has been placed so we don't need to compute it every turn
            if (
                not self.queen_placed[self.current_player_index]
                and tile is self.get_tile_by_id(self.current_player, "Queen", 1)
            ):
                self.queen_placed[self.current_player_index] = True
            # Handle 'covered' logic
            uncovered_tile = self.get_tile_at(old_pos)
            # if uncovered_tile is not None:
            #     print(f'moved {tile} from {old_pos} to {new_axial} and uncovered: {uncovered_tile}')
            if uncovered_tile is not None:
                uncovered_tile.covered = False
            if covered_tile is not None:
                covered_tile.covered = True
            
            self.end_turn()
            return True
        return False

    def update_all_valid_moves(self) -> None:
        """
        Update valid moves for all tiles.
        """
        # Get standard moves for each tile
        for tile in self.tiles:
            tile.update_valid_moves(self)

        # Get pillbug special moves
        self.get_pillbug_special_moves()

    def is_piece_surrounded(self, axial: Coordinate) -> bool:
        """
        Determine if the provided coordinate is surrounded by pieces. A piece is 'surrounded' if all 6 adjacent spaces are occupied by a tile.
        """
        if axial is None:
            return False  # An unplaced piece is not surrounded
        neighbours = self.get_adjacent_spaces(axial)
        for n in neighbours:
            if n[1] is None:
                return False
        return True

    def end_turn(self) -> None:
        """
        End the current player's turn and switch to the other player.
        """
        # print(f'end turn called, currently turn: {self.current_player}')
        # Increment the turn
        self.turn += 1

        # Update history and check for draw by repetition
        history_str = self.get_history_string()
        self.history.append(history_str)
        if self.history.count(history_str) == 3:
            return self.trigger_game_end("draw_r")

        # Check win conditions (is either/both queen surrounded)
        
        white_lost = self.is_piece_surrounded(self.get_tile_by_id("white", "Queen", 1).axial)
        black_lost = self.is_piece_surrounded(self.get_tile_by_id("black", "Queen", 1).axial)
        if white_lost and black_lost:
            return self.trigger_game_end("draw_q")
        elif white_lost:
            return self.trigger_game_end("black")
        elif black_lost:
            return self.trigger_game_end("white")

        # Pass turn to next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

        # Compute valid moves for this turn
        self.update_all_valid_moves()
        # print(">"*40)
        # print(self.current_player, self.print_valid_moves())
        # print("<"*40)

        # If there are no valid moves, switch back to other player
        if all([len(t.valid_moves) == 0 for t in self.tiles]):
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.update_all_valid_moves()
        
        # print(f'end turn finished, currently turn: {self.current_player}')

    def trigger_game_end(self, outcome_str: str) -> None:
        print(f" === GAME OVER, outcome: {outcome_str} ===")
        self.outcome = outcome_str
        self.update_all_valid_moves()
        # TODO other end of game cleanup
        # EG calculate a game score based on some heuristic - this would be important for AI to evaluate games

    def is_articulation_point(self, tile: Tile) -> bool:
        """
        Returns a boolean indicating if the provided tile is at an articulation point.

        An articulation point is a coord where removing the tile would disconnect the hive.
        """
        # An unplaced tile cannot be an articulation point
        if tile.axial is None:
            return False

        # Temporarily remove the tile from the board
        original_coord = tile.axial
        tile.axial = None

        # Build the set of coordinates for all placed tiles.
        hive_coords = {t.axial for t in self.tiles if t.axial is not None}
        if not hive_coords:
            # If there are no placed tiles (shouldn't happen), then it's not an articulation point.
            tile.axial = original_coord
            return False

        # Pick an arbitrary starting coordinate from the hive.
        start = next(iter(hive_coords))
        # Get all coordinates that are reachable from this starting point.
        reachable_coords = self.get_reachable_coords(start, set())

        # Restore the tile's position.
        tile.axial = original_coord

        # If the set of reachable coordinates is not equal to the full set of placed tile coordinates, then removing the tile disconnects the hive, making it an articulation point.
        return reachable_coords != hive_coords

    def get_reachable_coords(self, current: Coordinate, visited: set) -> set:
        """
        Recursively performs a depth-first search starting from 'current' to find all
        coordinates that are connected (via adjacent placed tiles) in the hive.

        :param current: The current coordinate being explored.
        :param visited: A set of coordinates already visited.
        :return: The complete set of reachable coordinates from the starting point.
        """
        visited.add(current)
        for neighbor_coord, neighbor_tile in self.get_adjacent_spaces(current):
            # Only consider neighbors that have a tile (i.e. are part of the hive)
            if neighbor_tile is not None and neighbor_coord not in visited:
                self.get_reachable_coords(neighbor_coord, visited)
        return visited

    def get_placements(self, color: str) -> Set[Coordinate]:
        """
        Get a list of all valid placements for a tile that is not yet placed on the board.
        """
        if self.outcome is not None:
            return set()
        
        # Turn 1 - if no tiles are placed, you may only place at (0,0)
        if self.turn == 0:
            return {(0,0)}

        # Turn 2 - if exactly one tile is placed, you must place adjacent to that tile
        elif self.turn == 1:
            return {s[0] for s in self.get_adjacent_spaces((0,0))}

        # Get all empty spaces adjacent to any of our own tiles.
        own_tiles = [t for t in self.tiles if t.color == color and t.axial is not None]
        potentials = set()
        for tile in own_tiles:
            for coord, adj_tile in self.get_adjacent_spaces(tile.axial):
                if adj_tile is None:
                    potentials.add(coord)

        # Filter out any potential placement that is adjacent to an enemy tile.
        placements = set()
        for coord in potentials:
            if any(t is not None and t.color != color for _, t in self.get_adjacent_spaces(coord)):
                continue
            placements.add(coord)

        return placements

    def get_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Return a list of axial coordinates, representing the valid moves for the provided tile.
        """
        if (
            self.outcome is not None
            or tile.covered  # A covered piece cannot move
            or tile is self.last_move  # A piece cannot be moved on two consecutive 'turns'
            or self.is_articulation_point(tile)  # You cannot move a piece if it is at an articulation point (moving it would break the hive into disconnected pieces)
        ):
            return set()
            
        # If the tile is a mosquito it copies the movement pattern of one of it's adjacent tiles.
        movement_pattern = {tile.tile_type}
        if tile.tile_type == 'Mosquito':
            movement_pattern.update({t.tile_type for c, t in self.get_adjacent_spaces(tile.axial) if t is not None})
        movement_pattern.discard('Mosquito')
            
        # TODO handle pillbug moves

        valid_moves = set()
        if tile.color == self.current_player:
            # It's your own piece, so check specific movement rules for the tile
            if 'Queen' in movement_pattern:
                valid_moves.update(self.get_queen_moves(tile))
            if 'Spider' in movement_pattern:
                valid_moves.update(self.get_spider_moves(tile))
            if 'Ant' in movement_pattern:
                valid_moves.update(self.get_ant_moves(tile))
            if 'Beetle' in movement_pattern:
                valid_moves.update(self.get_beetle_moves(tile))
            if 'Grasshopper' in movement_pattern:
                valid_moves.update(self.get_grasshopper_moves(tile))
            if 'Ladybug' in movement_pattern:
                valid_moves.update(self.get_ladybug_moves(tile))
            if 'Pillbug' in movement_pattern:
                valid_moves.update(self.get_pillbug_moves(tile))
                
        # Remove the starting tile coordinate - you can't finish a move at the coordinate where you started
        valid_moves.discard(tile.axial)

        return valid_moves

    def get_queen_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid queen moves for the current tile.
        
        The queen moves by crawling one space. This function is also
        """
        moves = set()
        
        # Ensure the tile is placed
        if tile.axial is None:
            return moves
        
        for c, t in self.get_adjacent_spaces(tile.axial):
            if self.try_crawl(tile.axial, c, tile.height):
                moves.add(c)
        return moves

    def get_spider_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid spider moves for the current tile.
        
        The spider moves by crawling exactly three spaces, with no back-tracking.
        """
        # Ensure the tile is placed
        if tile.axial is None:
            return set()

        # 'Lift' the tile off the board so it doesn't interfere with movement rules
        tmp_axial = tile.axial
        tile.axial = None
        
        # Spider makes exactly three moves
        frontier = {tmp_axial}
        visited = {tmp_axial}
        for _ in range(3):
            new_frontier = set()
            for f in frontier:
                for adj, _ in self.get_adjacent_spaces(f):
                    if adj not in visited and self.try_crawl(f, adj, tile.height):
                        new_frontier.add(adj)
            visited |= new_frontier
            frontier = new_frontier
            
        tile.axial = tmp_axial
        return frontier
    
    def get_ant_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid ant moves for the current tile.
    
        The ant moves by crawling one or more spaces (performing successive crawls) 
        without back-tracking. This implementation lifts the ant off the board,
        then flood-fills all reachable empty coordinates (using try_crawl) until no
        new positions are found. The starting coordinate is then removed from the
        final move set (since the ant cannot finish where it started).
        """
        # Ensure the tile is placed
        if tile.axial is None:
            return set()
    
        # Save and lift the ant off the board
        start = tile.axial
        tile.axial = None
    
        # Initialize the visited set with the starting coordinate
        # and set the frontier to start from that coordinate
        visited = {start}
        frontier = {start}
    
        # Perform a flood fill until no new moves are found
        while True:
            new_frontier = set()
            for pos in frontier:
                for adj, _ in self.get_adjacent_spaces(pos):
                    # Only consider positions we haven't already visited and that satisfy crawling rules
                    if adj not in visited and self.try_crawl(pos, adj, tile.height):
                        new_frontier.add(adj)
            if not new_frontier:
                break
            visited |= new_frontier
            frontier = new_frontier
    
        # Restore the ant's original position
        tile.axial = start
    
        # The ant cannot finish its move in the starting coordinate
        if start in visited:
            visited.remove(start)
    
        return visited

    def get_beetle_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid beetle moves for the current tile.

        The beetle moves to an adjacent tile either by crawl, climb, or fall. Note that it must observe the gate rule when crawling.
        """
        moves = set()
        
        # Ensure the tile is placed
        if tile.axial is None:
            return moves
            
        for c, t in self.get_adjacent_spaces(tile.axial):
            if (
                self.try_crawl(tile.axial, c, tile.height)
                or self.try_climb(tile.axial, c, tile.height, False)
                or self.try_fall(tile.axial, c, tile.height, False)
            ):
                moves.add(c)
        return moves

    def get_grasshopper_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid grasshopper moves for the current tile.

        The grasshopper moves by jumping over an adjacent piece.
        """
        moves = set()
        
        # Ensure the tile is placed
        if tile.axial is None:
            return moves
        
        # For each of the six adjacent directions
        for c, t in self.get_adjacent_spaces(tile.axial):
            # Only consider a direction if there is an adjacent tile
            if t is None:
                continue
            # Determine the direction vector (dq, dr) from the tile's current position
            dq = c[0] - tile.axial[0]
            dr = c[1] - tile.axial[1]
            # Starting from the neighbour, find the first empty space in that direction
            i = 1
            axial = (c[0] + dq, c[1] + dr)
            while self.get_tile_at(axial) is not None:
                i += 1
                axial = (c[0] + i * dq, c[1] + i * dr)
                # Safety check to prevent an infinite loop in extreme or erroneous cases.
                if i > 50:
                    raise ValueError(f"Error checking grasshopper move from {tile.axial} in direction {dq}, {dr}")
            moves.add(axial)

        return moves

    def get_pillbug_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid pillbug moves for the current tile.
        
        The pillbug moves by crawling one tile, the same as the queen.
        """
        return self.get_queen_moves(tile)

    def get_ladybug_moves(self, tile: Tile) -> Set[Coordinate]:
        """
        Get valid ladybug moves for the current tile using DFS.
        
        The ladybug must perform exactly three moves:
          1. A climb (from ground level only)
          2. A crawl or climb (without backtracking)
          3. A fall to an empty square
        
        This DFS implementation ensures that the visited set is local to each branch,
        so that overlapping intermediate cells in different chains do not block one another.
        """
        # Ladybug must be placed and on the ground.
        if tile.axial is None or tile.height > 0:
            return set()
    
        # Start DFS from the ladybug's current position at ground level (height 0).
        return self.ladybug_dfs(tile.axial, 0, moves_left=3, path=[tile.axial])

    def ladybug_dfs(self, pos: Coordinate, current_height: int, moves_left: int, path: List[Coordinate]) -> Set[Coordinate]:
        """
        Recursively explores move sequences for the ladybug.
        
        :param pos: The current coordinate.
        :param current_height: The current height of the ladybug.
        :param moves_left: The number of moves remaining (3 → 0).
        :param path: The list of coordinates visited in the current branch.
        :return: A set of destination coordinates reached at exactly 0 moves left.
        """
        # If no moves remain, we've completed a valid chain.
        if moves_left == 0:
            return {pos}
    
        results = set()
        
        # Phase selection based on how many moves remain:
        #   moves_left == 3: Phase 1 (climb only)
        #   moves_left == 2: Phase 2 (crawl OR climb)
        #   moves_left == 1: Phase 3 (fall only)
        if moves_left == 3:
            # Phase 1: Climb from ground level.
            for adj, _ in self.get_adjacent_spaces(pos):
                if adj in path:
                    continue
                if self.try_climb(pos, adj, current_height, one_step_only=False):
                    # For a climb, the new height becomes destination_tile.height + 1.
                    dest_tile = self.get_tile_at(adj)
                    if dest_tile is None:
                        continue
                    new_height = dest_tile.height + 1
                    results |= self.ladybug_dfs(adj, new_height, moves_left - 1, path + [adj])
        elif moves_left == 2:
            # Phase 2: Either crawl or climb.
            for adj, _ in self.get_adjacent_spaces(pos):
                if adj in path:
                    continue
                # Try crawling first (which does not change height).
                if self.try_crawl(pos, adj, current_height, validate_destination=True):
                    results |= self.ladybug_dfs(adj, current_height, moves_left - 1, path + [adj])
                # Then try climbing.
                if self.try_climb(pos, adj, current_height, one_step_only=False):
                    dest_tile = self.get_tile_at(adj)
                    if dest_tile is None:
                        continue
                    new_height = dest_tile.height + 1
                    results |= self.ladybug_dfs(adj, new_height, moves_left - 1, path + [adj])
        elif moves_left == 1:
            # Phase 3: Fall. The destination must be empty.
            for adj, tile_at_adj in self.get_adjacent_spaces(pos):
                if adj in path:
                    continue
                # For a fall, the destination must be empty and fall must be valid.
                if tile_at_adj is None and self.try_fall(pos, adj, current_height, one_step_only=False):
                    # For a valid fall (one-step fall), the ladybug lands on ground level.
                    results.add(adj)
        return results

    def get_pillbug_special_moves(self) -> None:
        """
        Based on the location of pillbug/mosquito tiles, identify and updated tiles that can be moved by the pillbug special rule.
        
        This special rule lets the pillbug (or mosquito adjacent to a pillbug) climb and adjacent piece onto itself, and then fall into an adjacent empty space.
        """
        # print(f'get_pillbug_special_moves called on turn {self.turn} for player {self.current_player}')
        if self.outcome is not None:
            return
        
        # Identify the pillbug(s) belonging to the current player
        my_pillbug_coords = set()
        for tile in self.tiles:
            if (
                tile.covered
                or tile.axial is None
                or tile is self.last_move
                or tile.color != self.current_player
            ):
                continue
            elif tile.tile_type == 'Pillbug':
                my_pillbug_coords.add(tile.axial)
            elif (
                tile.tile_type == 'Mosquito'
                and any([t.tile_type == 'Pillbug' for _, t in self.get_adjacent_spaces(tile.axial) if t is not None])
            ):
                my_pillbug_coords.add(tile.axial)
        # print(f'  my_pillbug_coords: {my_pillbug_coords}')
                
        for pb in my_pillbug_coords:
            valid_tiles_to_move = set()
            valid_destinations = set()
            for c, t in self.get_adjacent_spaces(pb):
                # print(f'  c: {c}, t: {t}')
                if t is None and self.try_fall(pb, c, 1, True):
                    valid_destinations.add(c)
                elif t is not None:
                    if (
                        t.covered
                        or t.height > 0
                        or t is self.last_move
                        or self.is_articulation_point(t)
                    ):
                        continue
                    elif self.try_climb(c, pb, 0, True):
                        valid_tiles_to_move.add(t)
            for tile in valid_tiles_to_move:
                tile.valid_moves.update(valid_destinations)

    def try_fall(self, source: Coordinate, destination: Coordinate, starting_height: int, one_step_only: bool = True) -> bool:
        """
        A fall is a two-step move. First perform a crawl to the destination coordinate, then decrease height.
        """
        #print(f"try_fall: from {source} to {destination} from height {starting_height}")
        if not starting_height > 0:
            return False
            
        # The destination must be empty or at a lower height than the starting height
        destination_tile = self.get_tile_at(destination)
        if destination_tile is not None and not destination_tile.height < starting_height:
            return False
        
        # Ensure you can crawl along the starting height to the destination tile
        if not self.try_crawl(source, destination, starting_height, False):
            return False
        
        # The destination tile must be at the correct height for a valid one-step fall
        if one_step_only:
            if (
                (destination_tile is None and starting_height != 1)
                or (destination_tile is not None and destination_tile.height != starting_height - 1)
            ):
                return False
           
        return True

    def try_climb(self, source: Coordinate, destination: Coordinate, starting_height: int, one_step_only: bool = True) -> bool:
        """
        A climb is a two-step move. First increase height, then perform a crawl to the destination square at the new height.
        
        A gate at `starting_height + 1` prevents a climb, even if you are climbing more than one step.
        """
        
        # The destination tile must be at the correct height for a valid climb
        destination_tile = self.get_tile_at(destination)
        if (
            destination_tile is None
            or destination_tile.height < starting_height
            or (one_step_only and destination_tile.height > starting_height)
        ):
            return False
        
        return self.try_crawl(source, destination, starting_height + 1, False)

    def try_crawl(self, source: Coordinate, destination: Coordinate, height: int, validate_destination: bool = True) -> bool:
        """
        A crawl is a move to an adjacent square at the same height to the starting coordinate.
        
        You cannot crawl through a 'gate'.
        """
        #print(f"try_crawl: {source} -> {destination} at h={height}")
        # By making this validation optional, we can reuse this function for other movement type (EG climbs)
        if validate_destination:
            # Destination must be same height as source
            destination_tile = self.get_tile_at(destination)
            # Ground-level move: destination must be empty
            if height == 0:
                if destination_tile is not None:
                    return False
            else:
                # Above-ground move: destination must be occupied by a tile exactly one level lower
                if destination_tile is None or destination_tile.height != (height - 1):
                    return False
        
        # Check that the source and destination coordinates are adjacent
        source_neighbour_coords = [s[0] for s in self.get_adjacent_spaces(source)]
        if destination not in source_neighbour_coords:
            return False
        
        # Find the two common neighbours for the source and desintation coordinates
        common_neighbours = []
        for c, t in self.get_adjacent_spaces(destination):
            if c in source_neighbour_coords:
                common_neighbours.append(t)
        
        # Make sure there are two common neighbours
        if len(common_neighbours) != 2:
            raise Exception("There must always be 2 common neighbours between any two adjacent tiles.")
        
        # Get neighbour heights
        l_height = 0 if common_neighbours[0] is None else common_neighbours[0].height
        r_height = 0 if common_neighbours[1] is None else common_neighbours[1].height
        if (
            common_neighbours[0] is not None
            and common_neighbours[1] is not None
            and l_height >= height
            and r_height >= height
        ):
            return False  # The common neighbours form a 'gate', preventing movement between them
        elif (
            height == 0
            and common_neighbours[0] is None
            and common_neighbours[1] is None
        ):
            return False  # The tile cannot lose contact with the hive while moving.
            
        return True

    def get_history_string(self) -> str:
        """
        Flatten the current game state into a string to record history.

        For each tile, we build a representation that includes:
          - color
          - tile_type
          - tile_id
          - axial coordinate (if placed) or None"" if unplaced"
          - height

        Each tile is represented as:
        {color}{tile_type}{tile_id}_{q,r or None}_{height}
        We then sort all these representations (using a sort key based on the tile's attributes)
        to ensure that the ordering is deterministic, regardless of the order in self.tiles.
        Finally, the sorted strings are concatenated (using "|" as a delimiter) to form the final history string.
        """
        pieces = []
        for tile in self.tiles:
            # If the tile is not placed, represent its coordinate as None"""
            if tile.axial is None:
                coord_str = "None"
                # Use a placeholder coordinate for sorting unplaced pieces.
                sort_coord = (-999, -999)
            else:
                q, r = tile.axial
                coord_str = f"{q},{r}"
                sort_coord = tile.axial

            # Build a string representation for this tile.
            piece_repr = f"{tile.color[0]}{tile.tile_type[0]}{tile.tile_id}_{coord_str}_{tile.height}"
            # Create a sort key tuple to ensure deterministic ordering.
            sort_key = (tile.color[0], tile.tile_type[0], tile.tile_id, sort_coord, tile.height)
            pieces.append((sort_key, piece_repr))

            # Sort by the sort key.
            pieces.sort(key=lambda x: x[0])

        # Join all piece representations into one history string.
        history_str = "|".join(piece_repr for _, piece_repr in pieces)
        return history_str

    def initialize_game(self) -> None:
        """
        Initializes the game state for a new game.

        This method resets the board state, adds all the initial tiles
        (the bag"" of pieces that have not yet been played), and records"
        the initial game history for turn 0.

        Modify the initial_tiles list to match the pieces and counts for your game.
        """
        # Reset game state
        self.tiles = []
        self.last_move = None
        self.history = []
        self.queen_placed = [False, False]
        self.current_player_index = 0
        self.turn = 0
        self.outcome = None


        # Define the initial pieces for each player.
        initial_tiles = [
            # White pieces
            ("white", "Queen", 1),
            ("white", "Spider", 1),
            ("white", "Spider", 2),
            ("white", "Beetle", 1),
            ("white", "Beetle", 2),
            ("white", "Grasshopper", 1),
            ("white", "Grasshopper", 2),
            ("white", "Ant", 1),
            ("white", "Ant", 2),
            ("white", "Ant", 3),
            # White expansion pieces
            ("white", "Ladybug", 1),
            ("white", "Mosquito", 1),
            ("white", "Pillbug", 1),
            # Black pieces
            ("black", "Queen", 1),
            ("black", "Spider", 1),
            ("black", "Spider", 2),
            ("black", "Beetle", 1),
            ("black", "Beetle", 2),
            ("black", "Grasshopper", 1),
            ("black", "Grasshopper", 2),
            ("black", "Ant", 1),
            ("black", "Ant", 2),
            ("black", "Ant", 3),
            # Black expansion pieces
            ("black", "Ladybug", 1),
            ("black", "Mosquito", 1),
            ("black", "Pillbug", 1),
        ]

        # Add each tile to the game state (tiles are unplaced, so axial remains None)
        for color, tile_type, tile_id in initial_tiles:
            tile = Tile(color, tile_type, tile_id)
            self.add_tile(tile)

        # Record the initial game state to the history.
        # This history string represents turn 0 (before any moves).
        initial_history = self.get_history_string()
        self.history.append(initial_history)

        # Set initial moves
        self.update_all_valid_moves()

    def print_valid_moves(self) -> str:
        """
        Helper function to return the current game state and all valid moves for each tile.
        """
        moves_str = ""
        for t in self.tiles:
            moves_str += f'{t} : {t.valid_moves}\n  '
        return(f'Game state: {self.history[-1]}\n  Current player: {self.current_player}\n  {moves_str}')

    def visualise_board(self) -> None:
        """
        Prints a crude ASCII representation of the game board.
        
        - Occupied cells display a symbol representing the tile (e.g. the first letter of its type).
        - Empty cells display the coordinate values.
        
        All cells are printed with a fixed width so that the board remains aligned.
        """
        # Get all placed tiles.
        placed_tiles = [tile for tile in self.tiles if tile.axial is not None]
        
        if placed_tiles:
            min_q = min(tile.axial[0] for tile in placed_tiles) - 1
            max_q = max(tile.axial[0] for tile in placed_tiles) + 1
            min_r = min(tile.axial[1] for tile in placed_tiles) - 1
            max_r = max(tile.axial[1] for tile in placed_tiles) + 1
        else:
            # Default board range when no tiles are placed.
            min_q, max_q, min_r, max_r = -3, 3, -3, 3
    
        # Build a dictionary for quick lookup.
        board = {tile.axial: tile for tile in placed_tiles}
        
        # Define a fixed cell width.
        cell_width = 9  # Adjust as needed
        
        # Helper function to format an empty cell with its coordinate.
        def format_empty(coord: Coordinate) -> str:
            # Format the coordinate as "q,r" and center it within the cell.
            coord_str = f"{coord[0]},{coord[1]}"
            return f"[{coord_str.center(cell_width - 2)}]"
    
        # Helper function to format an occupied cell.
        def format_tile(tile: Tile) -> str:
            # Use the first letter of the tile_type as a symbol.
            symbol = tile.tile_type[0] if tile.tile_type else "?"
            # For example, uppercase for white, lowercase for black.
            symbol = symbol.upper() if tile.color.lower() == "white" else symbol.lower()
            return f"[{symbol.center(cell_width - 2)}]"
    
        # Print the board row-by-row. We'll treat the r coordinate as rows.
        for r in range(min_r, max_r + 1):
            # Create an offset to mimic a hex grid (adjust as needed)
            offset = " " * 3 * (max_r - r)
            row_str = offset
            for q in range(min_q, max_q + 1):
                coord = (q, r)
                if coord in board:
                    cell = format_tile(board[coord])
                else:
                    cell = format_empty(coord)
                row_str += cell + " "
            print(row_str)

    def clone(self):
        """
        Create and return a deep copy of the game state.
        This allows simulations to run without modifying the original state.
        """
        return copy.deepcopy(self)

    def evaluate_state(self, me: str) -> int:
        """
        Return a numerical score for the game state from the perspective of current_player.
        Higher scores are better.
        """
        
        # TODO this needs to be more sophisticated, both evaluating the other players options and your own options by toggling the current player
        
        # Check for terminal states first.
        if self.outcome is not None:
            if self.outcome == me:
                return 10000
            elif self.outcome in ('draw_q', 'draw_r'):
                return 0
            else:
                return -10000
    
        score = 0
        
        # Evaluate how this move changed my position
        if self.current_player != me:
            self.set_current_player(me)
            self.update_all_valid_moves()
            
        # 1. Penalise pieces surrounding/covering my queen
        my_queen = self.get_tile_by_id(me, "Queen", 1)
        if my_queen.axial is not None:
            if self.get_tile_at(my_queen.axial).color != me:
                score -= 100
            for c, t in self.get_adjacent_spaces(my_queen.axial):
                if t is not None:
                    score -= 20 if t.color == me else 50
                    
        my_placed_tiles = [t for t in self.tiles if t.color == me and t.axial is not None]
        score += sum([-20 if len(t.valid_moves) == 0 else 20 for t in my_placed_tiles])

        # Evaluate how this move changed my opponents position
        them = "black" if me == "white" else "white"
        self.set_current_player(them)
        self.update_all_valid_moves()
        
        # 1. Reward pieces surrounding/covering opponent queen
        their_queen = self.get_tile_by_id(them, "Queen", 1)
        if their_queen.axial is not None:
            if self.get_tile_at(their_queen.axial).color == me:
                score += 100
            for c, t in self.get_adjacent_spaces(their_queen.axial):
                if t is not None:
                    score += 50 if t.color == me else 5
                    
            # 2. Penalise valid moves for their queen
            score -= 100 * len(their_queen.valid_moves)

        return score
    
    
    def choose_best_move(self) -> Tuple[str, str, int, Coordinate, int]:
        """
        Evaluate all legal moves for the current game state.
        Returns the move (tile, new_coord) with the highest score.
        """
        color = None
        tile_type = None
        tile_id = None
        move_coord = None
        best_score = float("-inf")
        choose_for_player = self.current_player
        
        for tile in self.tiles:
            for new_coord in tile.valid_moves:
                # Clone the game state so we don't disturb the actual game.
                sim_state = self.clone()
                sim_tile = sim_state.get_tile_by_id(tile.color, tile.tile_type, tile.tile_id)
                # Apply the move in the simulation.
                sim_state.move_tile(sim_tile, new_coord)
                # Evaluate the new state.
                score = sim_state.evaluate_state(choose_for_player)
                # Debug print.
                print(f"Evaluated move {sim_tile.tile_type} {sim_tile.tile_id} {sim_tile.axial} to {new_coord}: Score = {score}")
            
                if score > best_score:
                    color = sim_tile.color
                    tile_type = sim_tile.tile_type
                    tile_id = sim_tile.tile_id
                    move_coord = new_coord
                    best_score = score
    
        return color, tile_type, tile_id, move_coord, best_score
    
    def choose_weighted_move(self, weighted = True):
        legal_moves = []
        for tile in self.tiles:
            for move in tile.valid_moves:
                legal_moves.append((tile.color, tile.tile_type, tile.tile_id, move))
    
        eval_for_player = self.current_player    
        scores = []
        for move in legal_moves:
            color, tile_type, tile_id, new_coord = move
            sim_state = self.clone()
            sim_tile = sim_state.get_tile_by_id(color, tile_type, tile_id)
            
            sim_state.move_tile(sim_tile, new_coord)
            score = sim_state.evaluate_state(eval_for_player)
            scores.append(score)
            
        if weighted:
            # Convert scores to probabilities using softmax.
            exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
            probs = exp_scores / np.sum(exp_scores)
            # Choose a move randomly according to the probabilities.
            
            best_moves = np.argsort(probs)[-5:]
            for i in best_moves:
              print(f"prob: {probs[i]}, score: {scores[i]}, move: {legal_moves[i][0]} {legal_moves[i][1]} {legal_moves[i][2]} to {legal_moves[i][3]}")
            
            chosen_index = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[chosen_index], probs[chosen_index], scores[chosen_index]
        else:
            chosen_index = scores.index(max(scores))
            return legal_moves[chosen_index], 1, scores[chosen_index]
        

    def __repr__(self) -> str:
        return f"HiveGameState({len(self.tiles)} tiles, current player: {self.current_player})"


def command_line_interface(scenario: int):
    game_state = HiveGameState()
    game_state.initialize_game()

    print("Welcome to Hive!")
    print("Enter commands in the format:")
    print("    <color> <tile_type> <tile_id> to <q>,<r>")
    print("For example:")
    print("    white Queen 1 to 0,1")
    print("Type 'quit' to exit.\n")
    
    if scenario == 1:
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ant", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Grasshopper", 1), (0, -2))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ant", 2), (2,1))
    elif scenario == 2:
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ant", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Grasshopper", 1), (0, -2))
        game_state.move_tile(game_state.get_tile_by_id("black", "Grasshopper", 2), (0,2))
        game_state.move_tile(game_state.get_tile_by_id("white", "Grasshopper", 1), (0,3))
        game_state.move_tile(game_state.get_tile_by_id("black", "Beetle", 1), (2,1))
    elif scenario == 3:
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ant", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (-1, 2))
    elif scenario == 4:
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ant", 1), (-1, 2))
    elif scenario == 5:
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Beetle", 1), (-1, 2))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (0,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Beetle", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Beetle", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("white", "Grasshopper", 1), (0,-2))
        game_state.move_tile(game_state.get_tile_by_id("black", "Beetle", 1), (-1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (-1,1))
    elif scenario == 6:
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Grasshopper", 1), (0,2))
        game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 2), (1,-2))
        game_state.move_tile(game_state.get_tile_by_id("black", "Grasshopper", 1), (0,-2))
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 1), (1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 2), (0,2))
        game_state.move_tile(game_state.get_tile_by_id("white", "Ant", 2), (2,-2))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 2), (-1,0))
    elif scenario == 7:
        # Testing ladybug
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Ladybug", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Ladybug", 1), (0,2))
        # game_state.move_tile(game_state.get_tile_by_id("white", "Ladybug", 1), (1,0))
    elif scenario == 8:
        # Testing mosquitto
        game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
        game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
        game_state.move_tile(game_state.get_tile_by_id("white", "Pillbug", 1), (-1,-1))
        game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (0,2))
        
        
    outcome = None
    while outcome is None:
        # Print current state and valid moves
        print("\n" + "="*40)
        print("Current Turn:", game_state.turn)
        print("Current Player:", game_state.current_player)
        print("Outcome:", game_state.outcome)
        print(game_state.print_valid_moves())
        # print('')
        # game_state.visualise_board()
        # print('')
        
        foo = game_state.choose_best_move()
        print(foo)

        # Get input from the user
        command = input("Enter your command: ").strip()
        if command.lower() in ['quit', 'exit', 'q']:
            print("Exiting game.")
            break

        tokens = command.split()
        if len(tokens) < 5:
            print("Invalid command format. Please try again.")
            continue

        # Expected format: move <color> <tile_type> <tile_id> to <q>,<r>
        color = tokens[0]
        tile_type = tokens[1]
        tile_id = int(tokens[2])
        if tokens[3].lower() != "to":
            print("Expected keyword 'to' after tile id.")
            continue
        coord_tokens = tokens[4].split(',')
        if len(coord_tokens) != 2:
            print("Coordinate should be in format q,r (e.g. 0,1)")
            continue
        q = int(coord_tokens[0])
        r = int(coord_tokens[1])
        new_coord = (q, r)

        # Retrieve the specified tile
        tile = game_state.get_tile_by_id(color, tile_type, tile_id)
        if tile is None:
            print(f"Tile not found: {color} {tile_type} {tile_id}")
            continue

        # Attempt to move the tile (for unplaced tiles, this acts as placement)
        if new_coord not in tile.valid_moves:
            print(f"Invalid move for {tile}. Allowed moves: {tile.valid_moves}")
            continue

        success = game_state.move_tile(tile, new_coord)
        if success:
            print(f"Move successful: {tile} is now at {new_coord}")
        else:
            print("Move failed. Please try a different move.")


if __name__ == "__main__":
    # Create and interact with the game via command line.
    command_line_interface(8)
