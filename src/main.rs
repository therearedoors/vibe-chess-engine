use std::io::{self, Read};
use std::thread;
use serde::{Deserialize, Serialize};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use std::fmt;

#[derive(Clone, PartialEq)]
enum Color {
    White,
    Black,
}

#[derive(Clone, PartialEq)]
enum PieceType {
    Pawn, Knight, Bishop, Rook, Queen, King,
}

// Custom deserialization for Color
impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "white" => Ok(Color::White),
            "black" => Ok(Color::Black),
            _ => Err(serde::de::Error::custom(format!("Invalid color: {}", s))),
        }
    }
}

// Custom deserialization for PieceType
impl<'de> Deserialize<'de> for PieceType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "pawn" => Ok(PieceType::Pawn),
            "knight" => Ok(PieceType::Knight),
            "bishop" => Ok(PieceType::Bishop),
            "rook" => Ok(PieceType::Rook),
            "queen" => Ok(PieceType::Queen),
            "king" => Ok(PieceType::King),
            _ => Err(serde::de::Error::custom(format!("Invalid piece type: {}", s))),
        }
    }
}

#[derive(Deserialize, Clone)]
struct Piece {
    piece_type: PieceType,
    color: Color,
}

#[derive(Deserialize, Clone)]
struct GameState {
    board: Vec<Vec<Option<Piece>>>,
    current_turn: Color,
    move_history: Vec<Move>,
    player_side: Color,
    #[serde(default = "default_castling_rights")]
    castling_rights: CastlingRights,
    #[serde(default)]
    en_passant_target: Option<(i32, i32)>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Move {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    piece: String, // Matches "pawn", etc.
    captured: Option<String>,
    promotion: Option<String>,
}

#[derive(Deserialize, Clone)]
struct CastlingRights {
    #[serde(default = "default_true")]
    white_kingside: bool,
    #[serde(default = "default_true")]
    white_queenside: bool,
    #[serde(default = "default_true")]
    black_kingside: bool,
    #[serde(default = "default_true")]
    black_queenside: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ChessMove {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    promotion: Option<String>,
}

fn default_true() -> bool { true }

fn default_castling_rights() -> CastlingRights {
    CastlingRights {
        white_kingside: true,
        white_queenside: true,
        black_kingside: true,
        black_queenside: true,
    }
}

struct UndoInfo {
    captured: Option<Piece>,
    from_square: Option<Piece>,
    rook_from: Option<(usize, usize, Option<Piece>)>,
    en_passant_target: Option<(i32, i32)>,
    castling_rights: CastlingRights,
}

fn main() -> io::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    eprintln!("Received JSON: {}", buffer); // Debug print
    let game_state: GameState = serde_json::from_str(&buffer).expect("Failed to parse game state");

    let handle = thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
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
    let mut move_buffer = Vec::new();
    let depth = 2;
    let is_maximizing = game_state.current_turn == Color::White;
    let (_score, best_move) = minimax(&mut game_state, depth, -i32::MAX, i32::MAX, is_maximizing, &mut move_buffer);
    best_move.unwrap_or_else(|| ChessMove {
        from_row: 0,
        from_col: 0,
        to_row: 0,
        to_col: 0,
        promotion: None,
    })
}

fn minimax(game_state: &mut GameState, depth: i32, alpha: i32, beta: i32, maximizing: bool, move_buffer: &mut Vec<ChessMove>) -> (i32, Option<ChessMove>) {
    eprintln!("Minimax depth: {}", depth);
    if depth == 0 {
        return (quiescence_search(game_state, alpha, beta, maximizing, 2, move_buffer), None);
    }

    move_buffer.clear();
    move_buffer.extend(generate_legal_moves(game_state));
    if move_buffer.is_empty() {
        return (if maximizing { -100000 } else { 100000 }, None);
    }

    let moves = move_buffer.clone();
    let mut best_move = None;
    let original_turn = game_state.current_turn.clone();

    if maximizing {
        let mut max_eval = -i32::MAX;
        let mut alpha = alpha;
        for m in moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if is_in_check(game_state, &original_turn) {
                undo_move(game_state, m, undo);
                continue;
            }
            let (eval, _) = minimax(game_state, depth - 1, alpha, beta, false, move_buffer);
            if eval > max_eval {
                max_eval = eval;
                best_move = Some(m.clone());
            }
            undo_move(game_state, m, undo);
            alpha = alpha.max(eval);
            if beta <= alpha { break; }
        }
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        let mut beta = beta;
        for m in moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if is_in_check(game_state, &original_turn) {
                undo_move(game_state, m, undo);
                continue;
            }
            let (eval, _) = minimax(game_state, depth - 1, alpha, beta, true, move_buffer);
            if eval < min_eval {
                min_eval = eval;
                best_move = Some(m.clone());
            }
            undo_move(game_state, m, undo);
            beta = beta.min(eval);
            if beta <= alpha { break; }
        }
        (min_eval, best_move)
    }
}

fn quiescence_search(game_state: &mut GameState, mut alpha: i32, mut beta: i32, maximizing: bool, max_depth: i32, move_buffer: &mut Vec<ChessMove>) -> i32 {
    eprintln!("Quiescence depth: {}", max_depth);
    let stand_pat = evaluate_position(game_state);
    if !maximizing && stand_pat >= beta { return beta; }
    if maximizing && stand_pat < alpha { return alpha; }
    if maximizing { alpha = alpha.max(stand_pat); } else { beta = beta.min(stand_pat); }
    if max_depth <= 0 { return stand_pat; }

    move_buffer.clear();
    move_buffer.extend(generate_legal_moves(game_state).into_iter().filter(|m| game_state.board[m.to_row as usize][m.to_col as usize].is_some()));
    let moves = move_buffer.clone();

    let original_turn = game_state.current_turn.clone();
    if maximizing {
        for m in moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if is_in_check(game_state, &original_turn) {
                undo_move(game_state, m, undo);
                continue;
            }
            let score = quiescence_search(game_state, alpha, beta, false, max_depth - 1, move_buffer);
            alpha = alpha.max(score);
            undo_move(game_state, m, undo);
            if beta <= alpha { break; }
        }
        alpha
    } else {
        for m in moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if is_in_check(game_state, &original_turn) {
                undo_move(game_state, m, undo);
                continue;
            }
            let score = quiescence_search(game_state, alpha, beta, true, max_depth - 1, move_buffer);
            beta = beta.min(score);
            undo_move(game_state, m, undo);
            if beta <= alpha { break; }
        }
        beta
    }
}

fn generate_legal_moves(game_state: &GameState) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for from_row in 0..8 {
        for from_col in 0..8 {
            if let Some(piece) = &game_state.board[from_row][from_col] {
                if piece.color == game_state.current_turn {
                    match piece.piece_type {
                        PieceType::Pawn => moves.extend(generate_pawn_moves(game_state, from_row, from_col)),
                        PieceType::Knight => moves.extend(generate_knight_moves(game_state, from_row, from_col)),
                        PieceType::Bishop => moves.extend(generate_bishop_moves(game_state, from_row, from_col)),
                        PieceType::Rook => moves.extend(generate_rook_moves(game_state, from_row, from_col)),
                        PieceType::Queen => moves.extend(generate_queen_moves(game_state, from_row, from_col)),
                        PieceType::King => moves.extend(generate_king_moves(game_state, from_row, from_col)),
                    }
                }
            }
        }
    }
    moves.sort_by_key(|m| {
        let captured_value = game_state.board[m.to_row as usize][m.to_col as usize].as_ref().map(|p| match p.piece_type {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King => 20000,
        }).unwrap_or(0);
        let is_check = {
            let mut temp_state = game_state.clone();
            apply_move_in_place(&mut temp_state, m);
            is_in_check(&mut temp_state, if game_state.current_turn == Color::White { &Color::Black } else { &Color::White })
        };
        -(captured_value + (is_check as i32 * 1000))
    });
    moves
}

// Placeholder implementations (update these with enums as needed)
fn generate_pawn_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = game_state.board[row][col].as_ref().unwrap();
    let direction = if piece.color == Color::White { -1 } else { 1 };
    let start_row = if piece.color == Color::White { 6 } else { 1 };
    let promotion_row = if piece.color == Color::White { 0 } else { 7 };

    let to_row = (row as i32 + direction) as usize;
    if to_row < 8 && game_state.board[to_row][col].is_none() {
        if to_row == promotion_row {
            for promo in ["queen", "rook", "bishop", "knight"] {
                moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col: col as i32, promotion: Some(promo.to_string()) });
            }
        } else {
            moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row as i32, to_col: col as i32, promotion: None });
        }
        if row == start_row {
            let to_row2 = (row as i32 + 2 * direction) as usize;
            if game_state.board[to_row2][col].is_none() && game_state.board[to_row][col].is_none() {
                moves.push(ChessMove { from_row: row as i32, from_col: col as i32, to_row: to_row2 as i32, to_col: col as i32, promotion: None });
            }
        }
    }
    // Add captures, en passant (simplified)
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
}

fn generate_knight_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = game_state.board[row][col].as_ref().unwrap();
    let offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
    for (dr, dc) in offsets {
        let to_row = row as i32 + dr;
        let to_col = col as i32 + dc;
        if to_row >= 0 && to_row < 8 && to_col >= 0 && to_col < 8 {
            let target = &game_state.board[to_row as usize][to_col as usize];
            if target.is_none() || target.as_ref().unwrap().color != piece.color {
                moves.push(ChessMove {
                    from_row: row as i32,
                    from_col: col as i32,
                    to_row,
                    to_col,
                    promotion: None,
                });
            }
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
    if piece.color == Color::White && row == 7 && col == 4 {
        if game_state.castling_rights.white_kingside && game_state.board[7][5].is_none() && game_state.board[7][6].is_none() {
            moves.push(ChessMove { from_row: 7, from_col: 4, to_row: 7, to_col: 6, promotion: None });
        }
        if game_state.castling_rights.white_queenside && game_state.board[7][3].is_none() && game_state.board[7][2].is_none() && game_state.board[7][1].is_none() {
            moves.push(ChessMove { from_row: 7, from_col: 4, to_row: 7, to_col: 2, promotion: None });
        }
    } else if piece.color == Color::Black && row == 0 && col == 4 {
        if game_state.castling_rights.black_kingside && game_state.board[0][5].is_none() && game_state.board[0][6].is_none() {
            moves.push(ChessMove { from_row: 0, from_col: 4, to_row: 0, to_col: 6, promotion: None });
        }
        if game_state.castling_rights.black_queenside && game_state.board[0][3].is_none() && game_state.board[0][2].is_none() && game_state.board[0][1].is_none() {
            moves.push(ChessMove { from_row: 0, from_col: 4, to_row: 0, to_col: 2, promotion: None });
        }
    }
    moves.into_iter().filter(|m| is_legal_move(game_state, m)).collect()
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
    true
}

fn apply_move_in_place(game_state: &mut GameState, chess_move: &ChessMove) -> UndoInfo {
    let from_row = chess_move.from_row as usize;
    let from_col = chess_move.from_col as usize;
    let to_row = chess_move.to_row as usize;
    let to_col = chess_move.to_col as usize;

    let piece = game_state.board[from_row][from_col].as_ref().unwrap().clone();
    let captured = game_state.board[to_row][to_col].clone();
    let original_en_passant = game_state.en_passant_target;
    let original_castling = game_state.castling_rights.clone();

    game_state.board[to_row][to_col] = Some(piece.clone());
    game_state.board[from_row][from_col] = None;

    game_state.current_turn = match game_state.current_turn {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };

    UndoInfo {
        captured,
        from_square: Some(piece),
        rook_from: None, // Add castling logic if needed
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
    game_state.current_turn = match game_state.current_turn {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    game_state.en_passant_target = undo.en_passant_target;
    game_state.castling_rights = undo.castling_rights;
}

fn evaluate_position(game_state: &GameState) -> i32 {
    let mut score = 0;
    for row in &game_state.board {
        for square in row {
            if let Some(piece) = square {
                let value = match piece.piece_type {
                    PieceType::Pawn => 100,
                    PieceType::Knight => 320,
                    PieceType::Bishop => 330,
                    PieceType::Rook => 500,
                    PieceType::Queen => 900,
                    PieceType::King => 20000,
                };
                score += if piece.color == Color::White { value } else { -value };
            }
        }
    }
    score
}

fn is_in_check(game_state: &mut GameState, color: &Color) -> bool {
    let mut king_pos = None;
    for r in 0..8 {
        for c in 0..8 {
            if let Some(piece) = &game_state.board[r][c] {
                if piece.piece_type == PieceType::King && piece.color == *color {
                    king_pos = Some((r, c));
                    break;
                }
            }
        }
    }
    let (king_row, king_col) = king_pos.unwrap();

    let opponent_color = match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let original_turn = game_state.current_turn.clone();
    game_state.current_turn = opponent_color;
    let mut move_buffer = Vec::new();
    move_buffer.extend(generate_legal_moves(game_state));
    let result = move_buffer.iter().any(|m| m.to_row as usize == king_row && m.to_col as usize == king_col);
    game_state.current_turn = original_turn;
    result
}