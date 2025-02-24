import pygame
import math
from hive import HiveGameState  # Assuming your HiveGameState and related classes are in hive.py

# Pygame configuration
WIDTH, HEIGHT = 800, 600
HEX_SIZE = 40  # Adjust to scale your hex grid

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hive Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- Helper Rendering Functions ---
def axial_to_pixel(q, r, hex_size=HEX_SIZE):
    """Convert axial coordinates (q, r) to pixel coordinates."""
    x = hex_size * math.sqrt(3) * (q + r/2) + WIDTH/2
    y = hex_size * 3/2 * r + HEIGHT/2
    return (x, y)

def draw_hexagon(surface, center, size, color, width=0):
    """Draw a hexagon on the given surface."""
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        point = (center[0] + size * math.cos(angle_rad),
                 center[1] + size * math.sin(angle_rad))
        points.append(point)
    pygame.draw.polygon(surface, color, points, width)
    
def draw_board(game_state):
    """
    Draws hexagons only around placed tiles and their adjacent empty spaces.
    If no tiles are placed, it defaults to showing the (0,0) hex.
    """
    occupied_tiles = {tile.axial for tile in game_state.tiles if tile.axial is not None}
    adjacent_tiles = set()

    # Find all adjacent spaces around occupied tiles
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in occupied_tiles:
        for dq, dr in directions:
            neighbor = (q + dq, r + dr)
            if neighbor not in occupied_tiles:
                adjacent_tiles.add(neighbor)
                
    # If no tiles are placed, show the (0,0) hex
    if not occupied_tiles:
        occupied_tiles.add((0, 0))

    # Draw hexagons around both occupied and adjacent empty tiles
    for q, r in occupied_tiles | adjacent_tiles:
        center = axial_to_pixel(q, r)
        draw_hexagon(screen, center, HEX_SIZE, (200, 200, 200), 1)  # Outline for empty spaces

        # Draw coordinates inside hex
        coord_text = font.render(f"{q},{r}", True, (150, 150, 150))
        coord_rect = coord_text.get_rect(center=center)
        screen.blit(coord_text, coord_rect)


            
def draw_game_state(game_state):
    """
    Draw the tiles from the game state on the board.
    Each placed tile is drawn as a filled hexagon with:
    - The tile type's first letter
    - The tile number
    - The coordinate inside the hex
    """
    for tile in game_state.tiles:
        if tile.axial is not None and not tile.covered:
            center = axial_to_pixel(tile.axial[0], tile.axial[1])
            # Use blue for white pieces, red for black pieces.
            color = (0, 0, 255) if tile.color.lower() == "white" else (255, 0, 0)
            draw_hexagon(screen, center, HEX_SIZE, color, 0)  # Fill the hexagon

            # Draw the tile type's first letter (biggest text)
            text_tile = font.render(f'{tile.tile_type[0]}{tile.tile_id}', True, (255, 255, 255))
            text_rect = text_tile.get_rect(center=(center[0], center[1] - HEX_SIZE // 4))
            screen.blit(text_tile, text_rect)

            # Draw the tile number (middle-sized text)
            tile_number_str = f'h{tile.height}'  # Assuming tile has an ID number
            number_font = pygame.font.SysFont(None, 20)
            text_number = number_font.render(tile_number_str, True, (255, 255, 255))
            number_rect = text_number.get_rect(center=center)
            screen.blit(text_number, number_rect)

            # Draw the coordinate (smallest text)
            coord_str = f"{tile.axial[0]},{tile.axial[1]}"
            coord_font = pygame.font.SysFont(None, 16)
            text_coord = coord_font.render(coord_str, True, (255, 255, 255))
            coord_rect = text_coord.get_rect(center=(center[0], center[1] + HEX_SIZE // 4))
            screen.blit(text_coord, coord_rect)


def ai_move(game_state):
    print(f"AI move being selected")
    move = game_state.choose_weighted_move(True)
    if move is not None:
        m, prob, score = move
        color, tile_type, tile_id, new_coord = m
        t = game_state.get_tile_by_id(color, tile_type, tile_id)
        print(f"AI chose move: {t.tile_type} {t.tile_id} to {new_coord} with score {score}, prob {prob}")
        game_state.move_tile(t, new_coord)
    else:
        print("No legal moves for AI!")
    # print(1/0)

# --- Pygame-Based Input and Game Loop ---

def pygame_interface(scenario: int):
    # Initialize the game state from hive.py
    game_state = HiveGameState()
    game_state.initialize_game()

    # game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 1), (0,0))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Spider", 1), (0,1))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Queen", 1), (0, -1))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Queen", 1), (1,1))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Pillbug", 1), (-1,0))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (0,2))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (1,-1))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (1,2))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Spider", 2), (-2,1))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (0,2))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 1), (0,0))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (1,2))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 2), (-3,1))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (0,2))
    # game_state.move_tile(game_state.get_tile_by_id("white", "Beetle", 2), (-2,1))
    # game_state.move_tile(game_state.get_tile_by_id("black", "Pillbug", 1), (1,2))

    input_text = ""  # To hold the current command string

    running = True
    human_player = "white"
    ai_turn = human_player == "black"
    
    while running:
        if ai_turn:
            ai_move(game_state)
            if game_state.current_player == human_player:
                ai_turn = False
            if game_state.outcome is not None:
                print(1/0)
                
            # Clear screen
            screen.fill((255, 255, 255))
            # Draw the board and game state.
            draw_board(game_state)
            draw_game_state(game_state)
    
            # Draw current game state info (e.g., turn, current player)
            turn_info = font.render(f"Turn: {game_state.turn}   Player: {game_state.current_player}", True, (0, 0, 0))
            screen.blit(turn_info, (10, 10))
    
            # Draw the input command at the bottom of the screen.
            input_surface = font.render("Command: " + input_text, True, (0, 0, 0))
            screen.blit(input_surface, (10, 30))
    
            pygame.display.flip()
            clock.tick(30)
            
        else:
            while not ai_turn:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
        
                    elif event.type == pygame.KEYDOWN:
                        # When Enter is pressed, process the command
                        if event.key == pygame.K_RETURN:
                            command = input_text.strip()
                            if command.lower() in ['quit', 'exit', 'q']:
                                running = False
                                break
        
                            # Process the command string.
                            # Expected format: <color> <tile_type> <tile_id> to <q>,<r>
                            tokens = command.split()
                            if len(tokens) < 4:
                                print("Invalid command format. Please try again.")
                                input_text = ""
                                continue
        
                            color = tokens[0]
                            tile_type = tokens[1]
                            try:
                                tile_id = int(tokens[2])
                            except ValueError:
                                print("Tile id must be an integer.")
                                input_text = ""
                                continue
                            coord_tokens = tokens[3].split(',')
                            if len(coord_tokens) != 2:
                                print("Coordinate should be in format q,r (e.g. 0,1)")
                                input_text = ""
                                continue
                            try:
                                q = int(coord_tokens[0])
                                r = int(coord_tokens[1])
                            except ValueError:
                                print("Coordinate values must be integers.")
                                input_text = ""
                                continue
                            new_coord = (q, r)
        
                            # Retrieve the specified tile.
                            c = ("white" if color == "w" else ("black" if color == "b" else None))
                            t = None
                            if tile_type == "q":
                              t = "Queen"
                            elif tile_type == "a":
                              t = "Ant"
                            elif tile_type == "b":
                              t = "Beetle"
                            elif tile_type == "g":
                              t = "Grasshopper"
                            elif tile_type == "s":
                              t = "Spider"
                            elif tile_type == "l":
                              t = "Ladybug"
                            elif tile_type == "m":
                              t = "Mosquito"
                            elif tile_type == "p":
                              t = "Pillbug"
                            tile = game_state.get_tile_by_id(c, t, tile_id)
                            if tile is None:
                                print(f"Tile not found: {color} {tile_type} {tile_id}")
                                input_text = ""
                                continue
        
                            # Validate the move.
                            if new_coord not in tile.valid_moves:
                                print(f"Invalid move for {tile}. Allowed moves: {tile.valid_moves}")
                                input_text = ""
                                continue
        
                            success = game_state.move_tile(tile, new_coord)
                            if success:
                                print(f"Move successful: {tile} is now at {new_coord}")
                                if game_state.current_player != human_player:
                                    ai_turn = True
                            else:
                                print("Move failed. Please try a different move.")
                            # Clear the input after processing.
                            input_text = ""
        
                        # Handle backspace and regular characters.
                        elif event.key == pygame.K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            input_text += event.unicode
              
    
                # Clear screen
                screen.fill((255, 255, 255))
                # Draw the board and game state.
                draw_board(game_state)
                draw_game_state(game_state)
        
                # Draw current game state info (e.g., turn, current player)
                turn_info = font.render(f"Turn: {game_state.turn}   Player: {game_state.current_player}", True, (0, 0, 0))
                screen.blit(turn_info, (10, 10))
        
                # Draw the input command at the bottom of the screen.
                input_surface = font.render("Command: " + input_text, True, (0, 0, 0))
                screen.blit(input_surface, (10, 30))
        
                pygame.display.flip()
                clock.tick(30)
            

    pygame.quit()

if __name__ == "__main__":
    # For testing purposes, you can pass a scenario number if needed.
    pygame_interface(scenario=1)
