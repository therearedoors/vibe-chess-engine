use std::io::{self, Read};
use serde::{Deserialize, Serialize};

// Game state struct to receive from JS
#[derive(Deserialize)]
struct GameState {
    board: Vec<Vec<Option<Piece>>>, // 8x8 board, None for empty
    current_turn: String,          // "white" or "black"
    move_history: Vec<Move>,       // For castling/en passant
    player_side: String,           // "white" or "black" (AI's opponent)
}

// Piece representation
#[derive(Deserialize)]
struct Piece {
    piece_type: String, // "pawn", "rook", etc.
    color: String,      // "white" or "black"
}

// Move history entry
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Move {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    piece: String,
}

// Move output struct
#[derive(Serialize)]
struct ChessMove {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
}

fn main() -> io::Result<()> {
    // Read game state from stdin (JSON)
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let game_state: GameState = serde_json::from_str(&buffer).expect("Failed to parse game state");

    // Placeholder AI logic: pick a random move (replace with real logic later)
    let move_result = calculate_move(&game_state);

    // Output move as JSON to stdout
    let output = serde_json::to_string(&move_result).expect("Failed to serialize move");
    println!("{}", output);

    Ok(())
}

fn calculate_move(game_state: &GameState) -> ChessMove {
    // Dummy logic: move first piece found one square forward (for now)
    for row in 0..8 {
        for col in 0..8 {
            if let Some(piece) = &game_state.board[row][col] {
                if piece.color == game_state.current_turn {
                    let to_row = if piece.color == "white" { row - 1 } else { row + 1 };
                    if to_row >= 0 && to_row < 8 {
                        return ChessMove {
                            from_row: row as i32,
                            from_col: col as i32,
                            to_row: to_row as i32,
                            to_col: col as i32,
                        };
                    }
                }
            }
        }
    }
    // Fallback if no move found (shouldn’t happen with real logic)
    ChessMove {
        from_row: 0,
        from_col: 0,
        to_row: 0,
        to_col: 0,
    }
}

