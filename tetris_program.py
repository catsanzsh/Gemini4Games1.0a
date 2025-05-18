import pygame
import random
import sys
import math
import numpy as np

# --- Constants ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
FPS = 60

# Colors (approximating FlashBlox video)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_NAVY_BLUE = (15, 25, 80)
MID_NAVY_BLUE = (25, 40, 100)
LIGHT_GREY = (192, 192, 192)
PALE_YELLOW = (250, 240, 190)
STAR_COLORS = [(255, 255, 255), (255, 255, 200), (200, 200, 255)]

# Tetromino Colors
COLOR_I = (255, 0, 0)
COLOR_O = (0, 0, 255)
COLOR_T = (255, 255, 0)
COLOR_S = (0, 255, 0)
COLOR_Z = (100, 200, 100)
COLOR_J = (192, 192, 192)
COLOR_L = (255, 165, 0)
EMPTY_CELL_COLOR = DARK_NAVY_BLUE

# Board dimensions
BLOCK_SIZE = 18
BOARD_WIDTH_CELLS = 10
BOARD_HEIGHT_CELLS = 20
BOARD_AREA_WIDTH = BOARD_WIDTH_CELLS * BLOCK_SIZE
BOARD_AREA_HEIGHT = BOARD_HEIGHT_CELLS * BLOCK_SIZE

# Board position
BOARD_X_OFFSET = 40
BOARD_Y_OFFSET = (SCREEN_HEIGHT - BOARD_AREA_HEIGHT) // 2

# UI Panel
UI_X_OFFSET = BOARD_X_OFFSET + BOARD_AREA_WIDTH + 30
UI_Y_OFFSET = BOARD_Y_OFFSET
UI_WIDTH = SCREEN_WIDTH - UI_X_OFFSET - 30
UI_HEIGHT = BOARD_AREA_HEIGHT
UI_BG_COLOR = MID_NAVY_BLUE
UI_BORDER_COLOR = (50, 70, 130)
UI_TEXT_COLOR = WHITE

# Tetromino shapes
SHAPES_DATA = {
    'I': {'shape': [[1, 1, 1, 1]], 'color': COLOR_I},
    'O': {'shape': [[1, 1], [1, 1]], 'color': COLOR_O},
    'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': COLOR_T},
    'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': COLOR_S},
    'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': COLOR_Z},
    'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': COLOR_J},
    'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': COLOR_L}
}
SHAPE_KEYS = list(SHAPES_DATA.keys())

# Game Boy Tetris fall speeds (frames per grid cell drop)
LEVEL_FALL_SPEEDS = [
    48, 43, 38, 33, 28, 23, 18, 13, 8, 6,
    5, 5, 5, 4, 4, 4, 3, 3, 3, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 1
]
MAX_LEVEL = len(LEVEL_FALL_SPEEDS) - 1

# --- Sound Engine (Game Boy Style, No External Files) ---
def generate_square_wave(freq, duration, volume=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sign(np.sin(2 * np.pi * freq * t)) * volume
    # Create stereo array: duplicate mono signal for left and right channels
    stereo_wave = np.column_stack((wave, wave))
    sound_array = (stereo_wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)

def generate_noise(duration, volume=0.05, sample_rate=44100):
    noise = np.random.normal(0, volume, int(sample_rate * duration))
    # Create stereo array
    stereo_noise = np.column_stack((noise, noise))
    sound_array = (stereo_noise * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)

# Initialize Pygame mixer for stereo
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Sound effects
SOUND_MOVE = generate_square_wave(440, 0.05)  # Short blip for movement
SOUND_ROTATE = generate_square_wave(880, 0.05)  # Higher blip for rotation
SOUND_LOCK = generate_square_wave(220, 0.1)  # Lower tone for locking
SOUND_LINE_CLEAR = generate_square_wave(660, 0.15)  # Triumphant note
SOUND_GAME_OVER = generate_noise(0.5)  # Harsh noise for game over

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris4k by CATSEEK R1.")
clock = pygame.time.Clock()
DEFAULT_FONT_SMALL = pygame.font.Font(None, 28)
DEFAULT_FONT_MEDIUM = pygame.font.Font(None, 36)
DEFAULT_FONT_LARGE = pygame.font.Font(None, 48)
FLASHBLOX_FONT = pygame.font.Font(None, 55)

# --- Game Variables ---
game_state = "start"
board = []
current_piece = None
next_piece = None
score = 0
level = 0
lines_cleared_total = 0
fall_tick_counter = 0
stars = []

def generate_stars(num_stars):
    s = []
    for _ in range(num_stars):
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        size = random.randint(1, 2)
        color = random.choice(STAR_COLORS)
        s.append({'x': x, 'y': y, 'size': size, 'color': color})
    return s

stars = generate_stars(100)

# --- Helper Functions ---
def draw_text(surface, text, font, color, x, y, center_x=False, center_y=False):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center_x:
        text_rect.centerx = x
    else:
        text_rect.x = x
    if center_y:
        text_rect.centery = y
    else:
        text_rect.y = y
    surface.blit(text_surface, text_rect)
    return text_rect

def create_board():
    return [[EMPTY_CELL_COLOR for _ in range(BOARD_WIDTH_CELLS)] for _ in range(BOARD_HEIGHT_CELLS)]

def get_new_piece():
    shape_key = random.choice(SHAPE_KEYS)
    p_data = SHAPES_DATA[shape_key]
    return {
        'type': shape_key,
        'shape': p_data['shape'],
        'color': p_data['color'],
        'x': BOARD_WIDTH_CELLS // 2 - len(p_data['shape'][0]) // 2,
        'y': 0
    }

def rotate_piece_matrix(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

def check_collision(board_state, piece_matrix, piece_x, piece_y):
    for r_idx, row in enumerate(piece_matrix):
        for c_idx, cell in enumerate(row):
            if cell:
                board_r, board_c = piece_y + r_idx, piece_x + c_idx
                if not (0 <= board_c < BOARD_WIDTH_CELLS and 0 <= board_r < BOARD_HEIGHT_CELLS):
                    return True
                if board_state[board_r][board_c] != EMPTY_CELL_COLOR:
                    return True
    return False

def lock_piece(board_state, piece):
    global score
    for r_idx, row in enumerate(piece['shape']):
        for c_idx, cell in enumerate(row):
            if cell:
                board_r, board_c = piece['y'] + r_idx, piece['x'] + c_idx
                if 0 <= board_r < BOARD_HEIGHT_CELLS and 0 <= board_c < BOARD_WIDTH_CELLS:
                    board_state[board_r][board_c] = piece['color']
    SOUND_LOCK.play()

def clear_lines(board_state):
    global lines_cleared_total, score, level
    lines_to_clear = 0
    new_board = [row[:] for row in board_state]
    
    r = BOARD_HEIGHT_CELLS - 1
    while r >= 0:
        if all(cell != EMPTY_CELL_COLOR for cell in new_board[r]):
            lines_to_clear += 1
            del new_board[r]
            new_board.insert(0, [EMPTY_CELL_COLOR for _ in range(BOARD_WIDTH_CELLS)])
        else:
            r -= 1
            
    if lines_to_clear > 0:
        lines_cleared_total += lines_to_clear
        if lines_to_clear == 1:
            score += 40 * (level + 1)
        elif lines_to_clear == 2:
            score += 100 * (level + 1)
        elif lines_to_clear == 3:
            score += 300 * (level + 1)
        elif lines_to_clear == 4:
            score += 1200 * (level + 1)
        SOUND_LINE_CLEAR.play()
        new_level = lines_cleared_total // 10
        if new_level > level and new_level <= MAX_LEVEL:
            level = new_level
        elif new_level > MAX_LEVEL:
            level = MAX_LEVEL

    return new_board, lines_to_clear

def reset_game():
    global board, current_piece, next_piece, score, level, lines_cleared_total, fall_tick_counter, game_state
    board = create_board()
    current_piece = get_new_piece()
    next_piece = get_new_piece()
    score = 0
    level = 0
    lines_cleared_total = 0
    fall_tick_counter = 0
    game_state = "playing"

# --- Drawing Functions ---
def draw_stars_and_moon(surface):
    pygame.draw.circle(surface, PALE_YELLOW, (BOARD_X_OFFSET - 10, BOARD_Y_OFFSET + 60), 25)
    pygame.draw.circle(surface, DARK_NAVY_BLUE, (BOARD_X_OFFSET + 5, BOARD_Y_OFFSET + 50), 20)
    for star in stars:
        pygame.draw.rect(surface, star['color'], (star['x'], star['y'], star['size'], star['size']))

def draw_city_silhouette(surface):
    pygame.draw.ellipse(surface, BLACK, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT - 100, 120, 80))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 - 10, SCREEN_HEIGHT - 100 - 30),
        (SCREEN_WIDTH // 2 + 10, SCREEN_HEIGHT - 100 - 30),
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100 - 50)
    ])
    pygame.draw.rect(surface, BLACK, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 80, 20, 60))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 - 90, SCREEN_HEIGHT - 80 - 30)
    ])
    pygame.draw.rect(surface, BLACK, (SCREEN_WIDTH // 2 + 80, SCREEN_HEIGHT - 80, 20, 60))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 + 80, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 + 90, SCREEN_HEIGHT - 80 - 30)
    ])
    pygame.draw.rect(surface, BLACK, (0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40))

def draw_board_and_pieces(surface, board_state, piece_to_draw):
    pygame.draw.rect(surface, (5, 10, 40), (BOARD_X_OFFSET, BOARD_Y_OFFSET, BOARD_AREA_WIDTH, BOARD_AREA_HEIGHT))
    for r_idx, row in enumerate(board_state):
        for c_idx, cell_color in enumerate(row):
            if cell_color != EMPTY_CELL_COLOR:
                pygame.draw.rect(surface, cell_color,
                                 (BOARD_X_OFFSET + c_idx * BLOCK_SIZE,
                                  BOARD_Y_OFFSET + r_idx * BLOCK_SIZE,
                                  BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(surface, BLACK,
                                 (BOARD_X_OFFSET + c_idx * BLOCK_SIZE,
                                  BOARD_Y_OFFSET + r_idx * BLOCK_SIZE,
                                  BLOCK_SIZE, BLOCK_SIZE), 1)
    if piece_to_draw:
        for r_idx, row_data in enumerate(piece_to_draw['shape']):
            for c_idx, cell in enumerate(row_data):
                if cell:
                    pygame.draw.rect(surface, piece_to_draw['color'],
                                     (BOARD_X_OFFSET + (piece_to_draw['x'] + c_idx) * BLOCK_SIZE,
                                      BOARD_Y_OFFSET + (piece_to_draw['y'] + r_idx) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, BLACK,
                                     (BOARD_X_OFFSET + (piece_to_draw['x'] + c_idx) * BLOCK_SIZE,
                                      BOARD_Y_OFFSET + (piece_to_draw['y'] + r_idx) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 1)
    for r in range(BOARD_HEIGHT_CELLS + 1):
        pygame.draw.line(surface, (40, 40, 90),
                         (BOARD_X_OFFSET, BOARD_Y_OFFSET + r * BLOCK_SIZE),
                         (BOARD_X_OFFSET + BOARD_AREA_WIDTH, BOARD_Y_OFFSET + r * BLOCK_SIZE))
    for c in range(BOARD_WIDTH_CELLS + 1):
        pygame.draw.line(surface, (40, 40, 90),
                         (BOARD_X_OFFSET + c * BLOCK_SIZE, BOARD_Y_OFFSET),
                         (BOARD_X_OFFSET + c * BLOCK_SIZE, BOARD_Y_OFFSET + BOARD_AREA_HEIGHT))

def draw_ui(surface, current_score, current_level, lines, next_p):
    pygame.draw.rect(surface, UI_BG_COLOR, (UI_X_OFFSET, UI_Y_OFFSET, UI_WIDTH, UI_HEIGHT))
    pygame.draw.rect(surface, UI_BORDER_COLOR, (UI_X_OFFSET, UI_Y_OFFSET, UI_WIDTH, UI_HEIGHT), 3)
    padding = 20
    line_height = 35
    current_y = UI_Y_OFFSET + padding
    draw_text(surface, "score", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(current_score), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(current_score))[0], current_y)
    current_y += line_height + 10
    draw_text(surface, "level", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(current_level), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(current_level))[0], current_y)
    current_y += line_height + 10
    draw_text(surface, "lines", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(lines), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(lines))[0], current_y)
    current_y += line_height + 30
    draw_text(surface, "next", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    current_y += line_height - 10
    preview_area_width = 4 * BLOCK_SIZE
    preview_area_height = 4 * BLOCK_SIZE
    preview_x = UI_X_OFFSET + (UI_WIDTH - preview_area_width) // 2
    preview_y = current_y
    pygame.draw.rect(surface, UI_BORDER_COLOR, (preview_x - 5, preview_y - 5, preview_area_width + 10, preview_area_height + 10))
    pygame.draw.rect(surface, (5, 10, 40), (preview_x, preview_y, preview_area_width, preview_area_height))
    if next_p:
        shape_matrix = next_p['shape']
        shape_w = len(shape_matrix[0]) * BLOCK_SIZE
        shape_h = len(shape_matrix) * BLOCK_SIZE
        start_x = preview_x + (preview_area_width - shape_w) // 2
        start_y = preview_y + (preview_area_height - shape_h) // 2
        for r_idx, row in enumerate(shape_matrix):
            for c_idx, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(surface, next_p['color'],
                                     (start_x + c_idx * BLOCK_SIZE,
                                      start_y + r_idx * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, BLACK,
                                     (start_x + c_idx * BLOCK_SIZE,
                                      start_y + r_idx * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_start_screen(surface):
    surface.fill(DARK_NAVY_BLUE)
    draw_stars_and_moon(surface)
    draw_city_silhouette(surface)
    title_color = (255, 180, 0)
    draw_text(surface, "TETRIS4K", FLASHBLOX_FONT, title_color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60, center_x=True)
    draw_text(surface, "TM", DEFAULT_FONT_SMALL, title_color, SCREEN_WIDTH // 2 + FLASHBLOX_FONT.size("TETRIS4K")[0]//2 + 5, SCREEN_HEIGHT // 2 - 70)
    button_center_x = SCREEN_WIDTH // 2
    button_center_y = SCREEN_HEIGHT // 2 + 30
    button_radius = 30
    start_button_rect = pygame.Rect(button_center_x - button_radius, button_center_y - button_radius, 2 * button_radius, 2 * button_radius)
    pygame.draw.circle(surface, (0, 200, 0), (button_center_x, button_center_y), button_radius)
    pygame.draw.polygon(surface, WHITE, [
        (button_center_x - 10, button_center_y - 15),
        (button_center_x - 10, button_center_y + 15),
        (button_center_x + 15, button_center_y)
    ])
    draw_text(surface, "START", DEFAULT_FONT_MEDIUM, WHITE, button_center_x, button_center_y + button_radius + 25, center_x=True)
    return start_button_rect

def draw_game_over_screen(surface, final_score):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    draw_text(surface, "GAME OVER", DEFAULT_FONT_LARGE, (255, 50, 50), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60, center_x=True)
    draw_text(surface, f"Final Score: {final_score}", DEFAULT_FONT_MEDIUM, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10, center_x=True)
    play_again_rect = draw_text(surface, "play again", DEFAULT_FONT_MEDIUM, (150, 255, 150), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70, center_x=True)
    return play_again_rect

# --- Main Game Loop ---
start_button_clickable_area = None
play_again_clickable_area = None

while True:
    current_fall_speed_frames = LEVEL_FALL_SPEEDS[min(level, MAX_LEVEL)]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if game_state == "start":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and start_button_clickable_area and start_button_clickable_area.collidepoint(event.pos):
                    reset_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    reset_game()

        elif game_state == "playing":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece['x'] -= 1
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        current_piece['x'] += 1
                    else:
                        SOUND_MOVE.play()
                elif event.key == pygame.K_RIGHT:
                    current_piece['x'] += 1
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        current_piece['x'] -= 1
                    else:
                        SOUND_MOVE.play()
                elif event.key == pygame.K_DOWN:
                    fall_tick_counter = current_fall_speed_frames
                    score += 1
                    SOUND_MOVE.play()
                elif event.key == pygame.K_UP:
                    rotated_shape = rotate_piece_matrix(current_piece['shape'])
                    kick_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)]
                    if current_piece['type'] == 'I':
                        kick_offsets.extend([(0, -1), (0, -2), (0, 1)])
                    rotated_successfully = False
                    for dx, dy_kick in kick_offsets:
                        temp_x = current_piece['x'] + dx
                        temp_y = current_piece['y']
                        if not check_collision(board, rotated_shape, temp_x, temp_y):
                            current_piece['shape'] = rotated_shape
                            current_piece['x'] = temp_x
                            current_piece['y'] = temp_y
                            rotated_successfully = True
                            SOUND_ROTATE.play()
                            break
                elif event.key == pygame.K_SPACE:
                    while not check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y'] + 1):
                        current_piece['y'] += 1
                        score += 2
                    lock_piece(board, current_piece)
                    board, lines_just_cleared = clear_lines(board)
                    current_piece = next_piece
                    next_piece = get_new_piece()
                    fall_tick_counter = 0
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        game_state = "game_over"
                        SOUND_GAME_OVER.play()
                        screen.fill(DARK_NAVY_BLUE)
                        draw_stars_and_moon(screen)
                        draw_city_silhouette(screen)
                        draw_board_and_pieces(screen, board, current_piece)
                        draw_ui(screen, score, level, lines_cleared_total, next_piece)

        elif game_state == "game_over":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and play_again_clickable_area and play_again_clickable_area.collidepoint(event.pos):
                    reset_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    reset_game()

    if game_state == "playing":
        fall_tick_counter += 1
        if fall_tick_counter >= current_fall_speed_frames:
            fall_tick_counter = 0
            current_piece['y'] += 1
            if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                current_piece['y'] -= 1
                lock_piece(board, current_piece)
                board, lines_just_cleared = clear_lines(board)
                current_piece = next_piece
                next_piece = get_new_piece()
                if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                    game_state = "game_over"
                    SOUND_GAME_OVER.play()
                    screen.fill(DARK_NAVY_BLUE)
                    draw_stars_and_moon(screen)
                    draw_city_silhouette(screen)
                    draw_board_and_pieces(screen, board, current_piece)
                    draw_ui(screen, score, level, lines_cleared_total, next_piece)

    screen.fill(DARK_NAVY_BLUE)
    draw_stars_and_moon(screen)
    draw_city_silhouette(screen)

    if game_state == "start":
        start_button_clickable_area = draw_start_screen(screen)
    elif game_state == "playing":
        draw_board_and_pieces(screen, board, current_piece)
        draw_ui(screen, score, level, lines_cleared_total, next_piece)
    elif game_state == "game_over":
        draw_board_and_pieces(screen, board, None)
        draw_ui(screen, score, level, lines_cleared_total, next_piece)
        play_again_clickable_area = draw_game_over_screen(screen, score)

    pygame.display.flip()
    clock.tick(FPS)import pygame
import random
import sys
import math
import numpy as np

# --- Constants ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
FPS = 60

# Colors (approximating FlashBlox video)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_NAVY_BLUE = (15, 25, 80)
MID_NAVY_BLUE = (25, 40, 100)
LIGHT_GREY = (192, 192, 192)
PALE_YELLOW = (250, 240, 190)
STAR_COLORS = [(255, 255, 255), (255, 255, 200), (200, 200, 255)]

# Tetromino Colors
COLOR_I = (255, 0, 0)
COLOR_O = (0, 0, 255)
COLOR_T = (255, 255, 0)
COLOR_S = (0, 255, 0)
COLOR_Z = (100, 200, 100)
COLOR_J = (192, 192, 192)
COLOR_L = (255, 165, 0)
EMPTY_CELL_COLOR = DARK_NAVY_BLUE

# Board dimensions
BLOCK_SIZE = 18
BOARD_WIDTH_CELLS = 10
BOARD_HEIGHT_CELLS = 20
BOARD_AREA_WIDTH = BOARD_WIDTH_CELLS * BLOCK_SIZE
BOARD_AREA_HEIGHT = BOARD_HEIGHT_CELLS * BLOCK_SIZE

# Board position
BOARD_X_OFFSET = 40
BOARD_Y_OFFSET = (SCREEN_HEIGHT - BOARD_AREA_HEIGHT) // 2

# UI Panel
UI_X_OFFSET = BOARD_X_OFFSET + BOARD_AREA_WIDTH + 30
UI_Y_OFFSET = BOARD_Y_OFFSET
UI_WIDTH = SCREEN_WIDTH - UI_X_OFFSET - 30
UI_HEIGHT = BOARD_AREA_HEIGHT
UI_BG_COLOR = MID_NAVY_BLUE
UI_BORDER_COLOR = (50, 70, 130)
UI_TEXT_COLOR = WHITE

# Tetromino shapes
SHAPES_DATA = {
    'I': {'shape': [[1, 1, 1, 1]], 'color': COLOR_I},
    'O': {'shape': [[1, 1], [1, 1]], 'color': COLOR_O},
    'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': COLOR_T},
    'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': COLOR_S},
    'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': COLOR_Z},
    'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': COLOR_J},
    'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': COLOR_L}
}
SHAPE_KEYS = list(SHAPES_DATA.keys())

# Game Boy Tetris fall speeds (frames per grid cell drop)
LEVEL_FALL_SPEEDS = [
    48, 43, 38, 33, 28, 23, 18, 13, 8, 6,
    5, 5, 5, 4, 4, 4, 3, 3, 3, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 1
]
MAX_LEVEL = len(LEVEL_FALL_SPEEDS) - 1

# --- Sound Engine (Game Boy Style, No External Files) ---
def generate_square_wave(freq, duration, volume=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sign(np.sin(2 * np.pi * freq * t)) * volume
    # Create stereo array: duplicate mono signal for left and right channels
    stereo_wave = np.column_stack((wave, wave))
    sound_array = (stereo_wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)

def generate_noise(duration, volume=0.05, sample_rate=44100):
    noise = np.random.normal(0, volume, int(sample_rate * duration))
    # Create stereo array
    stereo_noise = np.column_stack((noise, noise))
    sound_array = (stereo_noise * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)

# Initialize Pygame mixer for stereo
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Sound effects
SOUND_MOVE = generate_square_wave(440, 0.05)  # Short blip for movement
SOUND_ROTATE = generate_square_wave(880, 0.05)  # Higher blip for rotation
SOUND_LOCK = generate_square_wave(220, 0.1)  # Lower tone for locking
SOUND_LINE_CLEAR = generate_square_wave(660, 0.15)  # Triumphant note
SOUND_GAME_OVER = generate_noise(0.5)  # Harsh noise for game over

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris4k by CATSEEK R1.")
clock = pygame.time.Clock()
DEFAULT_FONT_SMALL = pygame.font.Font(None, 28)
DEFAULT_FONT_MEDIUM = pygame.font.Font(None, 36)
DEFAULT_FONT_LARGE = pygame.font.Font(None, 48)
FLASHBLOX_FONT = pygame.font.Font(None, 55)

# --- Game Variables ---
game_state = "start"
board = []
current_piece = None
next_piece = None
score = 0
level = 0
lines_cleared_total = 0
fall_tick_counter = 0
stars = []

def generate_stars(num_stars):
    s = []
    for _ in range(num_stars):
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        size = random.randint(1, 2)
        color = random.choice(STAR_COLORS)
        s.append({'x': x, 'y': y, 'size': size, 'color': color})
    return s

stars = generate_stars(100)

# --- Helper Functions ---
def draw_text(surface, text, font, color, x, y, center_x=False, center_y=False):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center_x:
        text_rect.centerx = x
    else:
        text_rect.x = x
    if center_y:
        text_rect.centery = y
    else:
        text_rect.y = y
    surface.blit(text_surface, text_rect)
    return text_rect

def create_board():
    return [[EMPTY_CELL_COLOR for _ in range(BOARD_WIDTH_CELLS)] for _ in range(BOARD_HEIGHT_CELLS)]

def get_new_piece():
    shape_key = random.choice(SHAPE_KEYS)
    p_data = SHAPES_DATA[shape_key]
    return {
        'type': shape_key,
        'shape': p_data['shape'],
        'color': p_data['color'],
        'x': BOARD_WIDTH_CELLS // 2 - len(p_data['shape'][0]) // 2,
        'y': 0
    }

def rotate_piece_matrix(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

def check_collision(board_state, piece_matrix, piece_x, piece_y):
    for r_idx, row in enumerate(piece_matrix):
        for c_idx, cell in enumerate(row):
            if cell:
                board_r, board_c = piece_y + r_idx, piece_x + c_idx
                if not (0 <= board_c < BOARD_WIDTH_CELLS and 0 <= board_r < BOARD_HEIGHT_CELLS):
                    return True
                if board_state[board_r][board_c] != EMPTY_CELL_COLOR:
                    return True
    return False

def lock_piece(board_state, piece):
    global score
    for r_idx, row in enumerate(piece['shape']):
        for c_idx, cell in enumerate(row):
            if cell:
                board_r, board_c = piece['y'] + r_idx, piece['x'] + c_idx
                if 0 <= board_r < BOARD_HEIGHT_CELLS and 0 <= board_c < BOARD_WIDTH_CELLS:
                    board_state[board_r][board_c] = piece['color']
    SOUND_LOCK.play()

def clear_lines(board_state):
    global lines_cleared_total, score, level
    lines_to_clear = 0
    new_board = [row[:] for row in board_state]
    
    r = BOARD_HEIGHT_CELLS - 1
    while r >= 0:
        if all(cell != EMPTY_CELL_COLOR for cell in new_board[r]):
            lines_to_clear += 1
            del new_board[r]
            new_board.insert(0, [EMPTY_CELL_COLOR for _ in range(BOARD_WIDTH_CELLS)])
        else:
            r -= 1
            
    if lines_to_clear > 0:
        lines_cleared_total += lines_to_clear
        if lines_to_clear == 1:
            score += 40 * (level + 1)
        elif lines_to_clear == 2:
            score += 100 * (level + 1)
        elif lines_to_clear == 3:
            score += 300 * (level + 1)
        elif lines_to_clear == 4:
            score += 1200 * (level + 1)
        SOUND_LINE_CLEAR.play()
        new_level = lines_cleared_total // 10
        if new_level > level and new_level <= MAX_LEVEL:
            level = new_level
        elif new_level > MAX_LEVEL:
            level = MAX_LEVEL

    return new_board, lines_to_clear

def reset_game():
    global board, current_piece, next_piece, score, level, lines_cleared_total, fall_tick_counter, game_state
    board = create_board()
    current_piece = get_new_piece()
    next_piece = get_new_piece()
    score = 0
    level = 0
    lines_cleared_total = 0
    fall_tick_counter = 0
    game_state = "playing"

# --- Drawing Functions ---
def draw_stars_and_moon(surface):
    pygame.draw.circle(surface, PALE_YELLOW, (BOARD_X_OFFSET - 10, BOARD_Y_OFFSET + 60), 25)
    pygame.draw.circle(surface, DARK_NAVY_BLUE, (BOARD_X_OFFSET + 5, BOARD_Y_OFFSET + 50), 20)
    for star in stars:
        pygame.draw.rect(surface, star['color'], (star['x'], star['y'], star['size'], star['size']))

def draw_city_silhouette(surface):
    pygame.draw.ellipse(surface, BLACK, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT - 100, 120, 80))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 - 10, SCREEN_HEIGHT - 100 - 30),
        (SCREEN_WIDTH // 2 + 10, SCREEN_HEIGHT - 100 - 30),
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100 - 50)
    ])
    pygame.draw.rect(surface, BLACK, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 80, 20, 60))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 - 90, SCREEN_HEIGHT - 80 - 30)
    ])
    pygame.draw.rect(surface, BLACK, (SCREEN_WIDTH // 2 + 80, SCREEN_HEIGHT - 80, 20, 60))
    pygame.draw.polygon(surface, BLACK, [
        (SCREEN_WIDTH // 2 + 80, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT - 80 - 10),
        (SCREEN_WIDTH // 2 + 90, SCREEN_HEIGHT - 80 - 30)
    ])
    pygame.draw.rect(surface, BLACK, (0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40))

def draw_board_and_pieces(surface, board_state, piece_to_draw):
    pygame.draw.rect(surface, (5, 10, 40), (BOARD_X_OFFSET, BOARD_Y_OFFSET, BOARD_AREA_WIDTH, BOARD_AREA_HEIGHT))
    for r_idx, row in enumerate(board_state):
        for c_idx, cell_color in enumerate(row):
            if cell_color != EMPTY_CELL_COLOR:
                pygame.draw.rect(surface, cell_color,
                                 (BOARD_X_OFFSET + c_idx * BLOCK_SIZE,
                                  BOARD_Y_OFFSET + r_idx * BLOCK_SIZE,
                                  BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(surface, BLACK,
                                 (BOARD_X_OFFSET + c_idx * BLOCK_SIZE,
                                  BOARD_Y_OFFSET + r_idx * BLOCK_SIZE,
                                  BLOCK_SIZE, BLOCK_SIZE), 1)
    if piece_to_draw:
        for r_idx, row_data in enumerate(piece_to_draw['shape']):
            for c_idx, cell in enumerate(row_data):
                if cell:
                    pygame.draw.rect(surface, piece_to_draw['color'],
                                     (BOARD_X_OFFSET + (piece_to_draw['x'] + c_idx) * BLOCK_SIZE,
                                      BOARD_Y_OFFSET + (piece_to_draw['y'] + r_idx) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, BLACK,
                                     (BOARD_X_OFFSET + (piece_to_draw['x'] + c_idx) * BLOCK_SIZE,
                                      BOARD_Y_OFFSET + (piece_to_draw['y'] + r_idx) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 1)
    for r in range(BOARD_HEIGHT_CELLS + 1):
        pygame.draw.line(surface, (40, 40, 90),
                         (BOARD_X_OFFSET, BOARD_Y_OFFSET + r * BLOCK_SIZE),
                         (BOARD_X_OFFSET + BOARD_AREA_WIDTH, BOARD_Y_OFFSET + r * BLOCK_SIZE))
    for c in range(BOARD_WIDTH_CELLS + 1):
        pygame.draw.line(surface, (40, 40, 90),
                         (BOARD_X_OFFSET + c * BLOCK_SIZE, BOARD_Y_OFFSET),
                         (BOARD_X_OFFSET + c * BLOCK_SIZE, BOARD_Y_OFFSET + BOARD_AREA_HEIGHT))

def draw_ui(surface, current_score, current_level, lines, next_p):
    pygame.draw.rect(surface, UI_BG_COLOR, (UI_X_OFFSET, UI_Y_OFFSET, UI_WIDTH, UI_HEIGHT))
    pygame.draw.rect(surface, UI_BORDER_COLOR, (UI_X_OFFSET, UI_Y_OFFSET, UI_WIDTH, UI_HEIGHT), 3)
    padding = 20
    line_height = 35
    current_y = UI_Y_OFFSET + padding
    draw_text(surface, "score", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(current_score), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(current_score))[0], current_y)
    current_y += line_height + 10
    draw_text(surface, "level", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(current_level), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(current_level))[0], current_y)
    current_y += line_height + 10
    draw_text(surface, "lines", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    draw_text(surface, str(lines), DEFAULT_FONT_MEDIUM, UI_TEXT_COLOR, UI_X_OFFSET + UI_WIDTH - padding - DEFAULT_FONT_MEDIUM.size(str(lines))[0], current_y)
    current_y += line_height + 30
    draw_text(surface, "next", DEFAULT_FONT_SMALL, UI_TEXT_COLOR, UI_X_OFFSET + padding, current_y)
    current_y += line_height - 10
    preview_area_width = 4 * BLOCK_SIZE
    preview_area_height = 4 * BLOCK_SIZE
    preview_x = UI_X_OFFSET + (UI_WIDTH - preview_area_width) // 2
    preview_y = current_y
    pygame.draw.rect(surface, UI_BORDER_COLOR, (preview_x - 5, preview_y - 5, preview_area_width + 10, preview_area_height + 10))
    pygame.draw.rect(surface, (5, 10, 40), (preview_x, preview_y, preview_area_width, preview_area_height))
    if next_p:
        shape_matrix = next_p['shape']
        shape_w = len(shape_matrix[0]) * BLOCK_SIZE
        shape_h = len(shape_matrix) * BLOCK_SIZE
        start_x = preview_x + (preview_area_width - shape_w) // 2
        start_y = preview_y + (preview_area_height - shape_h) // 2
        for r_idx, row in enumerate(shape_matrix):
            for c_idx, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(surface, next_p['color'],
                                     (start_x + c_idx * BLOCK_SIZE,
                                      start_y + r_idx * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, BLACK,
                                     (start_x + c_idx * BLOCK_SIZE,
                                      start_y + r_idx * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_start_screen(surface):
    surface.fill(DARK_NAVY_BLUE)
    draw_stars_and_moon(surface)
    draw_city_silhouette(surface)
    title_color = (255, 180, 0)
    draw_text(surface, "TETRIS4K", FLASHBLOX_FONT, title_color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60, center_x=True)
    draw_text(surface, "TM", DEFAULT_FONT_SMALL, title_color, SCREEN_WIDTH // 2 + FLASHBLOX_FONT.size("TETRIS4K")[0]//2 + 5, SCREEN_HEIGHT // 2 - 70)
    button_center_x = SCREEN_WIDTH // 2
    button_center_y = SCREEN_HEIGHT // 2 + 30
    button_radius = 30
    start_button_rect = pygame.Rect(button_center_x - button_radius, button_center_y - button_radius, 2 * button_radius, 2 * button_radius)
    pygame.draw.circle(surface, (0, 200, 0), (button_center_x, button_center_y), button_radius)
    pygame.draw.polygon(surface, WHITE, [
        (button_center_x - 10, button_center_y - 15),
        (button_center_x - 10, button_center_y + 15),
        (button_center_x + 15, button_center_y)
    ])
    draw_text(surface, "START", DEFAULT_FONT_MEDIUM, WHITE, button_center_x, button_center_y + button_radius + 25, center_x=True)
    return start_button_rect

def draw_game_over_screen(surface, final_score):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    draw_text(surface, "GAME OVER", DEFAULT_FONT_LARGE, (255, 50, 50), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60, center_x=True)
    draw_text(surface, f"Final Score: {final_score}", DEFAULT_FONT_MEDIUM, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10, center_x=True)
    play_again_rect = draw_text(surface, "play again", DEFAULT_FONT_MEDIUM, (150, 255, 150), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70, center_x=True)
    return play_again_rect

# --- Main Game Loop ---
start_button_clickable_area = None
play_again_clickable_area = None

while True:
    current_fall_speed_frames = LEVEL_FALL_SPEEDS[min(level, MAX_LEVEL)]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if game_state == "start":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and start_button_clickable_area and start_button_clickable_area.collidepoint(event.pos):
                    reset_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    reset_game()

        elif game_state == "playing":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece['x'] -= 1
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        current_piece['x'] += 1
                    else:
                        SOUND_MOVE.play()
                elif event.key == pygame.K_RIGHT:
                    current_piece['x'] += 1
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        current_piece['x'] -= 1
                    else:
                        SOUND_MOVE.play()
                elif event.key == pygame.K_DOWN:
                    fall_tick_counter = current_fall_speed_frames
                    score += 1
                    SOUND_MOVE.play()
                elif event.key == pygame.K_UP:
                    rotated_shape = rotate_piece_matrix(current_piece['shape'])
                    kick_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)]
                    if current_piece['type'] == 'I':
                        kick_offsets.extend([(0, -1), (0, -2), (0, 1)])
                    rotated_successfully = False
                    for dx, dy_kick in kick_offsets:
                        temp_x = current_piece['x'] + dx
                        temp_y = current_piece['y']
                        if not check_collision(board, rotated_shape, temp_x, temp_y):
                            current_piece['shape'] = rotated_shape
                            current_piece['x'] = temp_x
                            current_piece['y'] = temp_y
                            rotated_successfully = True
                            SOUND_ROTATE.play()
                            break
                elif event.key == pygame.K_SPACE:
                    while not check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y'] + 1):
                        current_piece['y'] += 1
                        score += 2
                    lock_piece(board, current_piece)
                    board, lines_just_cleared = clear_lines(board)
                    current_piece = next_piece
                    next_piece = get_new_piece()
                    fall_tick_counter = 0
                    if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                        game_state = "game_over"
                        SOUND_GAME_OVER.play()
                        screen.fill(DARK_NAVY_BLUE)
                        draw_stars_and_moon(screen)
                        draw_city_silhouette(screen)
                        draw_board_and_pieces(screen, board, current_piece)
                        draw_ui(screen, score, level, lines_cleared_total, next_piece)

        elif game_state == "game_over":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and play_again_clickable_area and play_again_clickable_area.collidepoint(event.pos):
                    reset_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    reset_game()

    if game_state == "playing":
        fall_tick_counter += 1
        if fall_tick_counter >= current_fall_speed_frames:
            fall_tick_counter = 0
            current_piece['y'] += 1
            if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                current_piece['y'] -= 1
                lock_piece(board, current_piece)
                board, lines_just_cleared = clear_lines(board)
                current_piece = next_piece
                next_piece = get_new_piece()
                if check_collision(board, current_piece['shape'], current_piece['x'], current_piece['y']):
                    game_state = "game_over"
                    SOUND_GAME_OVER.play()
                    screen.fill(DARK_NAVY_BLUE)
                    draw_stars_and_moon(screen)
                    draw_city_silhouette(screen)
                    draw_board_and_pieces(screen, board, current_piece)
                    draw_ui(screen, score, level, lines_cleared_total, next_piece)

    screen.fill(DARK_NAVY_BLUE)
    draw_stars_and_moon(screen)
    draw_city_silhouette(screen)

    if game_state == "start":
        start_button_clickable_area = draw_start_screen(screen)
    elif game_state == "playing":
        draw_board_and_pieces(screen, board, current_piece)
        draw_ui(screen, score, level, lines_cleared_total, next_piece)
    elif game_state == "game_over":
        draw_board_and_pieces(screen, board, None)
        draw_ui(screen, score, level, lines_cleared_total, next_piece)
        play_again_clickable_area = draw_game_over_screen(screen, score)

    pygame.display.flip()
    clock.tick(FPS)
