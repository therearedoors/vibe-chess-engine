use std::io::{self, Read};
use std::thread;
use serde::{Deserialize, Serialize};

// Game state struct to receive from JS
#[derive(Deserialize, Clone)]
struct GameState {
    board: Vec<Vec<Option<Piece>>>,
    current_turn: String,
    move_history: Vec<Move>,
    player_side: String,
    castling_rights: CastlingRights,
    en_passant_target: Option<(i32, i32)>,   // Optional, defaults to None
}

#[derive(Deserialize, Clone)]
struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

// Piece representation
#[derive(Deserialize, Clone)] // Add Clone for easier handling
struct Piece {
    piece_type: String,
    color: String,
}

// Move history entry
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Move {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    piece: String,
    captured: Option<String>, // Track captured piece type
    promotion: Option<String>, // For pawn promotion
}

// Move output struct
#[derive(Serialize, Clone)]
struct ChessMove {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    promotion: Option<String>, // Add promotion to output
}

fn main() -> io::Result<()> {
    // Read game state from stdin (JSON)
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let game_state: GameState = serde_json::from_str(&buffer).expect("Failed to parse game state");

    let handle = thread::Builder::new()
    .stack_size(8 * 1024 * 1024) // 8MB stack
    .spawn(move || {
        let move_result = calculate_move(&game_state);
        let output = serde_json::to_string(&move_result).expect("Failed to serialize move");
        println!("{}", output);
    })?;
handle.join().unwrap();
Ok(())
}

fn calculate_move(game_state: &GameState) -> ChessMove {
    let mut game_state = game_state.clone();
    let depth = 2;
    let is_maximizing = game_state.current_turn == "white"; // Compute this first
    let (_score, best_move) = minimax(&mut game_state, depth, -i32::MAX, i32::MAX, is_maximizing);
    best_move.unwrap_or_else(|| ChessMove {
        from_row: 0,
        from_col: 0,
        to_row: 0,
        to_col: 0,
        promotion: None,
    })
}

fn is_legal_move(game_state: &GameState, chess_move: &ChessMove) -> bool {
    let from_row = chess_move.from_row as usize;
    let from_col = chess_move.from_col as usize;
    let to_row = chess_move.to_row as usize;
    let to_col = chess_move.to_col as usize;

    if from_row >= 8 || from_col >= 8 || to_row >= 8 || to_col >= 8 {
        return false;
    }

    let piece = match &game_state.board[from_row][from_col] {
        Some(p) => p,
        None => return false,
    };

    if piece.color != game_state.current_turn {
        return false;
    }

    let target = &game_state.board[to_row][to_col];
    if let Some(target_piece) = target {
        if target_piece.color == piece.color {
            return false;
        }
    }

    match piece.piece_type.as_str() {
        "pawn" => {
            let direction = if piece.color == "white" { -1 } else { 1 };
            let start_row = if piece.color == "white" { 6 } else { 1 };
            let row_diff = (to_row as i32 - from_row as i32) * direction;

            if to_col == from_col && row_diff == 1 && target.is_none() {
                return true;
            }
            if to_col == from_col && row_diff == 2 && from_row == start_row && target.is_none() && game_state.board[(from_row as i32 + direction) as usize][from_col].is_none() {
                return true;
            }
            if row_diff == 1 && (to_col as i32 - from_col as i32).abs() == 1 {
                if target.is_some() || (game_state.en_passant_target == Some((to_row as i32, to_col as i32))) {
                    return true;
                }
            }
            false
        }
        "knight" => {
            let row_diff = (to_row as i32 - from_row as i32).abs();
            let col_diff = (to_col as i32 - from_col as i32).abs();
            (row_diff == 2 && col_diff == 1) || (row_diff == 1 && col_diff == 2)
        }
        "bishop" => is_clear_diagonal(game_state, from_row, from_col, to_row, to_col),
        "rook" => is_clear_straight(game_state, from_row, from_col, to_row, to_col),
        "queen" => is_clear_diagonal(game_state, from_row, from_col, to_row, to_col) || is_clear_straight(game_state, from_row, from_col, to_row, to_col),
        "king" => {
            let row_diff = (to_row as i32 - from_row as i32).abs();
            let col_diff = (to_col as i32 - from_col as i32).abs();
            if row_diff <= 1 && col_diff <= 1 {
                return true;
            }
            // Castling
            if piece.color == "white" && from_row == 7 && from_col == 4 {
                if to_row == 7 && to_col == 6 && game_state.castling_rights.white_kingside {
                    // Kingside: check squares e1 (4), f1 (5), g1 (6)
                    if game_state.board[7][5].is_none() && game_state.board[7][6].is_none() && !is_in_check(game_state, "white") {
                        let mut temp_state = game_state.clone();
                        temp_state.board[7][4] = None; // Move king temporarily
                        temp_state.board[7][5] = Some(Piece { piece_type: "king".to_string(), color: "white".to_string() });
                        if !is_in_check(&temp_state, "white") {
                            temp_state.board[7][5] = None;
                            temp_state.board[7][6] = Some(Piece { piece_type: "king".to_string(), color: "white".to_string() });
                            return !is_in_check(&temp_state, "white");
                        }
                    }
                    return false;
                }
                if to_row == 7 && to_col == 2 && game_state.castling_rights.white_queenside {
                    // Queenside: check squares e1 (4), d1 (3), c1 (2)
                    if game_state.board[7][3].is_none() && game_state.board[7][2].is_none() && game_state.board[7][1].is_none() && !is_in_check(game_state, "white") {
                        let mut temp_state = game_state.clone();
                        temp_state.board[7][4] = None;
                        temp_state.board[7][3] = Some(Piece { piece_type: "king".to_string(), color: "white".to_string() });
                        if !is_in_check(&temp_state, "white") {
                            temp_state.board[7][3] = None;
                            temp_state.board[7][2] = Some(Piece { piece_type: "king".to_string(), color: "white".to_string() });
                            return !is_in_check(&temp_state, "white");
                        }
                    }
                    return false;
                }
            } else if piece.color == "black" && from_row == 0 && from_col == 4 {
                if to_row == 0 && to_col == 6 && game_state.castling_rights.black_kingside {
                    // Kingside: check squares e8 (4), f8 (5), g8 (6)
                    if game_state.board[0][5].is_none() && game_state.board[0][6].is_none() && !is_in_check(game_state, "black") {
                        let mut temp_state = game_state.clone();
                        temp_state.board[0][4] = None;
                        temp_state.board[0][5] = Some(Piece { piece_type: "king".to_string(), color: "black".to_string() });
                        if !is_in_check(&temp_state, "black") {
                            temp_state.board[0][5] = None;
                            temp_state.board[0][6] = Some(Piece { piece_type: "king".to_string(), color: "black".to_string() });
                            return !is_in_check(&temp_state, "black");
                        }
                    }
                    return false;
                }
                if to_row == 0 && to_col == 2 && game_state.castling_rights.black_queenside {
                    // Queenside: check squares e8 (4), d8 (3), c8 (2)
                    if game_state.board[0][3].is_none() && game_state.board[0][2].is_none() && game_state.board[0][1].is_none() && !is_in_check(game_state, "black") {
                        let mut temp_state = game_state.clone();
                        temp_state.board[0][4] = None;
                        temp_state.board[0][3] = Some(Piece { piece_type: "king".to_string(), color: "black".to_string() });
                        if !is_in_check(&temp_state, "black") {
                            temp_state.board[0][3] = None;
                            temp_state.board[0][2] = Some(Piece { piece_type: "king".to_string(), color: "black".to_string() });
                            return !is_in_check(&temp_state, "black");
                        }
                    }
                    return false;
                }
            }
            false
        }
        _ => false,
    }
}

// Helper functions for sliding pieces
fn is_clear_straight(game_state: &GameState, from_row: usize, from_col: usize, to_row: usize, to_col: usize) -> bool {
    if from_row != to_row && from_col != to_col {
        return false;
    }
    let (start, end, fixed, is_row) = if from_row == to_row {
        (from_col.min(to_col), from_col.max(to_col), from_row, false)
    } else {
        (from_row.min(to_row), from_row.max(to_row), from_col, true)
    };
    for i in (start + 1)..end {
        if is_row {
            if game_state.board[i][fixed].is_some() {
                return false;
            }
        } else {
            if game_state.board[fixed][i].is_some() {
                return false;
            }
        }
    }
    true
}

fn is_clear_diagonal(game_state: &GameState, from_row: usize, from_col: usize, to_row: usize, to_col: usize) -> bool {
    let row_diff = to_row as i32 - from_row as i32;
    let col_diff = to_col as i32 - from_col as i32;
    if row_diff.abs() != col_diff.abs() {
        return false;
    }
    let steps = row_diff.abs() as usize;
    let row_step = row_diff.signum();
    let col_step = col_diff.signum();
    for i in 1..steps {
        let r = from_row as i32 + i as i32 * row_step;
        let c = from_col as i32 + i as i32 * col_step;
        if game_state.board[r as usize][c as usize].is_some() {
            return false;
        }
    }
    true
}

fn is_in_check(game_state: &GameState, color: &str) -> bool {
    // Find the king
    let mut king_pos = None;
    for r in 0..8 {
        for c in 0..8 {
            if let Some(piece) = &game_state.board[r][c] {
                if piece.piece_type == "king" && piece.color == color {
                    king_pos = Some((r, c));
                    break;
                }
            }
        }
    }
    let (king_row, king_col) = king_pos.unwrap();

    // Temporarily switch turns to generate opponent moves
    let opponent_color = if color == "white" { "black" } else { "white" };
    let temp_state = GameState {
        current_turn: opponent_color.to_string(),
        ..game_state.clone()
    };
    let opponent_moves = generate_legal_moves(&temp_state);

    // Check if any opponent move captures the king
    opponent_moves.iter().any(|m| m.to_row as usize == king_row && m.to_col as usize == king_col)
}

fn generate_legal_moves(game_state: &GameState) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for from_row in 0..8 {
        for from_col in 0..8 {
            if let Some(piece) = &game_state.board[from_row][from_col] {
                if piece.color == game_state.current_turn {
                    match piece.piece_type.as_str() {
                        "pawn" => moves.extend(generate_pawn_moves(game_state, from_row, from_col)),
                        "knight" => moves.extend(generate_knight_moves(game_state, from_row, from_col)),
                        "bishop" => moves.extend(generate_bishop_moves(game_state, from_row, from_col)),
                        "rook" => moves.extend(generate_rook_moves(game_state, from_row, from_col)),
                        "queen" => moves.extend(generate_queen_moves(game_state, from_row, from_col)),
                        "king" => moves.extend(generate_king_moves(game_state, from_row, from_col)),
                        _ => {}
                    }
                }
            }
        }
    }
    moves
}

fn generate_pawn_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = game_state.board[row][col].as_ref().unwrap();
    let direction = if piece.color == "white" { -1 } else { 1 };
    let start_row = if piece.color == "white" { 6 } else { 1 };
    let promotion_row = if piece.color == "white" { 0 } else { 7 };

    // One square forward
    let to_row = (row as i32 + direction) as usize;
    if to_row < 8 && game_state.board[to_row][col].is_none() {
        if to_row == promotion_row {
            for promo in &["queen", "rook", "bishop", "knight"] {
                moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col: col as i32, promotion: Some(promo.to_string()) });
            }
        } else {
            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col: col as i32, promotion: None });
        }
        // Two squares from start
        if row == start_row {
            let to_row2 = (row as i32 + 2 * direction) as usize;
            if game_state.board[to_row2][col].is_none() && game_state.board[to_row][col].is_none() {
                moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row2 as i32, to_col: col as i32, promotion: None });
            }
        }
    }

    // Captures
    for dc in &[-1, 1] {
        let to_col = col as i32 + dc;
        if to_col >= 0 && to_col < 8 {
            let to_row = (row as i32 + direction) as usize;
            if let Some(target) = &game_state.board[to_row][to_col as usize] {
                if target.color != piece.color {
                    if to_row == promotion_row {
                        for promo in &["queen", "rook", "bishop", "knight"] {
                            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col, promotion: Some(promo.to_string()) });
                        }
                    } else {
                        moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col, promotion: None });
                    }
                }
            }
            // En passant
            if let Some((ep_row, ep_col)) = game_state.en_passant_target {
                if ep_row == to_row as i32 && ep_col == to_col {
                    moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col, promotion: None });
                }
            }
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn generate_knight_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
    let mut moves = Vec::new();
    for (dr, dc) in offsets {
        let to_row = row as i32 + dr;
        let to_col = col as i32 + dc;
        if to_row >= 0 && to_row < 8 && to_col >= 0 && to_col < 8 {
            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row, to_col, promotion: None });
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn generate_bishop_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for (dr, dc) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: r, to_col: c, promotion: None });
            if game_state.board[r as usize][c as usize].is_some() { break; }
            r += dr;
            c += dc;
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn generate_rook_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for (dr, dc) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: r, to_col: c, promotion: None });
            if game_state.board[r as usize][c as usize].is_some() { break; }
            r += dr;
            c += dc;
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn generate_queen_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = generate_bishop_moves(game_state, row, col);
    moves.extend(generate_rook_moves(game_state, row, col));
    moves
}

fn generate_king_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 { continue; }
            let to_row = row as i32 + dr;
            let to_col = col as i32 + dc;
            if to_row >= 0 && to_row < 8 && to_col >= 0 && to_col < 8 {
                moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row, to_col, promotion: None });
            }
        }
    }
    // Castling
    let piece = game_state.board[row][col].as_ref().unwrap();
    if piece.color == "white" && row == 7 && col == 4 {
        if game_state.castling_rights.white_kingside && game_state.board[7][5].is_none() && game_state.board[7][6].is_none() {
            moves.push(ChessMove { from_row: 7, from_col: 4, to_row: 7, to_col: 6, promotion: None });
        }
        if game_state.castling_rights.white_queenside && game_state.board[7][3].is_none() && game_state.board[7][2].is_none() && game_state.board[7][1].is_none() {
            moves.push(ChessMove { from_row: 7, from_col: 4, to_row: 7, to_col: 2, promotion: None });
        }
    } else if piece.color == "black" && row == 0 && col == 4 {
        if game_state.castling_rights.black_kingside && game_state.board[0][5].is_none() && game_state.board[0][6].is_none() {
            moves.push(ChessMove { from_row: 0, from_col: 4, to_row: 0, to_col: 6, promotion: None });
        }
        if game_state.castling_rights.black_queenside && game_state.board[0][3].is_none() && game_state.board[0][2].is_none() && game_state.board[0][1].is_none() {
            moves.push(ChessMove { from_row: 0, from_col: 4, to_row: 0, to_col: 2, promotion: None });
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn evaluate_position(game_state: &GameState) -> i32 {
    let mut score = 0;

    // Material values
    let piece_values = [
        ("pawn", 100),
        ("knight", 320),
        ("bishop", 330),
        ("rook", 500),
        ("queen", 900),
        ("king", 20000),
    ];

    // Positional bonuses
    let center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]; // d4, d5, e4, e5
    let king_safety_penalty = -10; // Per exposed square around king

    for row in 0..8 {
        for col in 0..8 {
            if let Some(piece) = &game_state.board[row][col] {
                let value = piece_values.iter()
                    .find(|&&(t, _)| t == piece.piece_type)
                    .map(|&(_, v)| v)
                    .unwrap_or(0);
                score += if piece.color == "white" { value } else { -value };

                // Positional bonuses
                match piece.piece_type.as_str() {
                    "pawn" => {
                        // Center control
                        if center_squares.contains(&(row, col)) {
                            score += if piece.color == "white" { 10 } else { -10 };
                        }
                        // Pawn advancement (closer to promotion)
                        let advance = if piece.color == "white" { 7 - row } else { row } as i32;
                        score += if piece.color == "white" { advance * 5 } else { -advance * 5 };
                    }
                    "knight" | "bishop" => {
                        // Mobility bonus: count legal moves
                        let temp_state = GameState { current_turn: piece.color.clone(), ..game_state.clone() };
                        let moves = match piece.piece_type.as_str() {
                            "knight" => generate_knight_moves(&temp_state, row, col),
                            "bishop" => generate_bishop_moves(&temp_state, row, col),
                            _ => vec![],
                        };
                        score += if piece.color == "white" { moves.len() as i32 * 5 } else { -(moves.len() as i32 * 5) };
                    }
                    "king" => {
                        // King safety: penalize exposed squares
                        for dr in -1..=1 {
                            for dc in -1..=1 {
                                let r = row as i32 + dr;
                                let c = col as i32 + dc;
                                if r >= 0 && r < 8 && c >= 0 && c < 8 && game_state.board[r as usize][c as usize].is_none() {
                                    score += if piece.color == "white" { king_safety_penalty } else { -king_safety_penalty };
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    score
}

fn minimax(game_state: &mut GameState, depth: i32, alpha: i32, beta: i32, maximizing: bool) -> (i32, Option<ChessMove>) {
    if depth == 0 {
        return (quiescence_search(game_state, alpha, beta, maximizing, 4), None);
    }

    let moves = generate_legal_moves(game_state);
    if moves.is_empty() {
        return (if maximizing { -100000 } else { 100000 }, None);
    }

    let mut best_move = None;
    let original_turn = game_state.current_turn.clone();

    if maximizing {
        let mut max_eval = -i32::MAX;
        let mut alpha = alpha;
        for m in moves {
            let undo = apply_move_in_place(game_state, &m);
            if is_in_check(game_state, original_turn.as_str()) {
                undo_move(game_state, &m, undo);
                continue;
            }
            let (eval, _) = minimax(game_state, depth - 1, alpha, beta, false);
            if eval > max_eval {
                max_eval = eval;
                best_move = Some(m.clone());
            }
            alpha = alpha.max(eval);
            undo_move(game_state, &m, undo);
            if beta <= alpha { break; }
        }
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        let mut beta = beta;
        for m in moves {
            let undo = apply_move_in_place(game_state, &m);
            if is_in_check(game_state, original_turn.as_str()) {
                undo_move(game_state, &m, undo);
                continue;
            }
            let (eval, _) = minimax(game_state, depth - 1, alpha, beta, true);
            if eval < min_eval {
                min_eval = eval;
                best_move = Some(m.clone());
            }
            beta = beta.min(eval);
            undo_move(game_state, &m, undo);
            if beta <= alpha { break; }
        }
        (min_eval, best_move)
    }
}

fn quiescence_search(game_state: &mut GameState, mut alpha: i32, mut beta: i32, maximizing: bool, max_depth: i32) -> i32 {
    let stand_pat = evaluate_position(game_state);
    if !maximizing && stand_pat >= beta { return beta; }
    if maximizing && stand_pat < alpha { return alpha; }
    if maximizing { alpha = alpha.max(stand_pat); } else { beta = beta.min(stand_pat); }
    if max_depth <= 0 { return stand_pat; } // Stop recursion here

    let moves = generate_legal_moves(game_state)
        .into_iter()
        .filter(|m| game_state.board[m.to_row as usize][m.to_col as usize].is_some()) // Captures only
        .collect::<Vec<_>>();

    let original_turn = game_state.current_turn.clone();
    if maximizing {
        for m in moves {
            let undo = apply_move_in_place(game_state, &m);
            if is_in_check(game_state, original_turn.as_str()) {
                undo_move(game_state, &m, undo);
                continue;
            }
            let score = quiescence_search(game_state, alpha, beta, false, max_depth - 1);
            alpha = alpha.max(score);
            undo_move(game_state, &m, undo);
            if beta <= alpha { break; }
        }
        alpha
    } else {
        for m in moves {
            let undo = apply_move_in_place(game_state, &m);
            if is_in_check(game_state, original_turn.as_str()) {
                undo_move(game_state, &m, undo);
                continue;
            }
            let score = quiescence_search(game_state, alpha, beta, true, max_depth - 1);
            beta = beta.min(score);
            undo_move(game_state, &m, undo);
            if beta <= alpha { break; }
        }
        beta
    }
}

struct UndoInfo {
    captured: Option<Piece>,
    from_square: Option<Piece>,
    rook_from: Option<(usize, usize, Option<Piece>)>,
    en_passant_target: Option<(i32, i32)>,
    castling_rights: CastlingRights,
}

fn apply_move_in_place(game_state: &mut GameState, chess_move: &ChessMove) -> UndoInfo {
    let from_row = chess_move.from_row as usize;
    let from_col = chess_move.from_col as usize;
    let to_row = chess_move.to_row as usize;
    let to_col = chess_move.to_col as usize;

    let piece = game_state.board[from_row][from_col].as_ref().unwrap().clone();
    let captured = game_state.board[to_row][to_col].clone();
    let mut rook_from = None;
    let original_en_passant = game_state.en_passant_target;
    let original_castling = game_state.castling_rights.clone();

    if piece.piece_type == "pawn" && chess_move.promotion.is_some() {
        game_state.board[to_row][to_col] = Some(Piece {
            piece_type: chess_move.promotion.as_ref().unwrap().clone(),
            color: piece.color.clone(),
        });
    } else {
        game_state.board[to_row][to_col] = Some(piece.clone());
    }
    game_state.board[from_row][from_col] = None;

    if piece.piece_type == "pawn" && game_state.en_passant_target == Some((to_row as i32, to_col as i32)) {
        let captured_row = if piece.color == "white" { to_row + 1 } else { to_row - 1 };
        game_state.board[captured_row][to_col] = None;
    } else if piece.piece_type == "pawn" && (to_row as i32 - from_row as i32).abs() == 2 {
        game_state.en_passant_target = Some(((from_row + to_row) as i32 / 2, to_col as i32));
    } else {
        game_state.en_passant_target = None;
    }

    if piece.piece_type == "king" && (to_col as i32 - from_col as i32).abs() == 2 {
        let rook_from_col = if to_col > from_col { 7 } else { 0 };
        let rook_to_col = if to_col > from_col { 5 } else { 3 };
        rook_from = Some((from_row, rook_from_col, game_state.board[from_row][rook_from_col].clone()));
        game_state.board[from_row][rook_to_col] = game_state.board[from_row][rook_from_col].clone();
        game_state.board[from_row][rook_from_col] = None;
    }

    if piece.piece_type == "king" {
        if piece.color == "white" {
            game_state.castling_rights.white_kingside = false;
            game_state.castling_rights.white_queenside = false;
        } else {
            game_state.castling_rights.black_kingside = false;
            game_state.castling_rights.black_queenside = false;
        }
    } else if piece.piece_type == "rook" {
        if piece.color == "white" {
            if from_row == 7 && from_col == 7 { game_state.castling_rights.white_kingside = false; }
            if from_row == 7 && from_col == 0 { game_state.castling_rights.white_queenside = false; }
        } else {
            if from_row == 0 && from_col == 7 { game_state.castling_rights.black_kingside = false; }
            if from_row == 0 && from_col == 0 { game_state.castling_rights.black_queenside = false; }
        }
    }

    game_state.current_turn = if game_state.current_turn == "white" { "black".to_string() } else { "white".to_string() };
    game_state.move_history.push(Move {
        from_row: chess_move.from_row,
        from_col: chess_move.from_col,
        to_row: chess_move.to_row,
        to_col: chess_move.to_col,
        piece: piece.piece_type.clone(),
        captured: captured.as_ref().map(|p| p.piece_type.clone()),
        promotion: chess_move.promotion.clone(),
    });

    UndoInfo {
        captured,
        from_square: Some(piece),
        rook_from,
        en_passant_target: original_en_passant,
        castling_rights: original_castling,
    }
}

fn undo_move(game_state: &mut GameState, chess_move: &ChessMove, undo: UndoInfo) {
    let from_row = chess_move.from_row as usize;
    let from_col = chess_move.from_col as usize;
    let to_row = chess_move.to_row as usize;
    let to_col = chess_move.to_col as usize;

    game_state.board[from_row][from_col] = undo.from_square;
    game_state.board[to_row][to_col] = undo.captured;

    if let Some((r, c, rook_piece)) = undo.rook_from {
        game_state.board[r][c] = rook_piece;
        let rook_to_col = if to_col > from_col { 5 } else { 3 };
        game_state.board[r][rook_to_col] = None;
    }

    game_state.current_turn = if game_state.current_turn == "white" { "black".to_string() } else { "white".to_string() };
    game_state.move_history.pop();
    game_state.en_passant_target = undo.en_passant_target;
    game_state.castling_rights = undo.castling_rights;
}