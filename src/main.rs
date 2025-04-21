use std::io::{self, Read};
use std::thread;
use serde::{Deserialize, Serialize};
use serde::de::Deserializer;

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
    // Reduced depth to prevent stack overflow
    let depth = 2;
    let is_maximizing = game_state.current_turn == Color::White;
    
    let legal_moves = generate_legal_moves(&mut game_state);
    if legal_moves.is_empty() {
        // Return a default move if no legal moves are available
        return ChessMove {
            from_row: 0,
            from_col: 0,
            to_row: 0,
            to_col: 0,
            promotion: None,
        };
    }
    
    // Use a non-recursive move selection approach if recursion becomes a problem
    let (_score, best_move) = minimax(&mut game_state, depth, -i32::MAX, i32::MAX, is_maximizing, &mut move_buffer);
    best_move.unwrap_or_else(|| legal_moves[0].clone())
}

fn minimax(game_state: &mut GameState, depth: i32, alpha: i32, beta: i32, maximizing: bool, move_buffer: &mut Vec<ChessMove>) -> (i32, Option<ChessMove>) {
    // Limit depth to prevent stack overflow
    if depth == 0 {
        // Use simple evaluation instead of quiescence search to avoid deep recursion
        return (evaluate_position(game_state), None);
    }

    // Generate and sort legal moves (moves that don't leave the king in check)
    move_buffer.clear();
    let legal_moves = generate_legal_moves_with_check_validation(game_state);
    if legal_moves.is_empty() {
        // If no legal moves, return a score indicating checkmate or stalemate
        let current_turn = game_state.current_turn.clone();
        if is_in_check(game_state, &current_turn) {
            return (if maximizing { -100000 } else { 100000 }, None);
        } else {
            return (0, None); // Stalemate
        }
    }
    
    move_buffer.extend(legal_moves);
    let moves = sort_moves(game_state, move_buffer);
    let mut best_move = None;

    if maximizing {
        let mut max_eval = -i32::MAX;
        let mut alpha = alpha;
        for m in moves.iter() {
            let undo = apply_move_in_place(game_state, m);
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

#[allow(dead_code)]
// Simplified quiescence search with limited depth to avoid stack overflow
fn quiescence_search(game_state: &mut GameState, mut alpha: i32, mut beta: i32, maximizing: bool, depth: i32) -> i32 {
    // Hard limit on quiescence depth
    if depth <= 0 {
        return evaluate_position(game_state);
    }
    
    let stand_pat = evaluate_position(game_state);
    
    if maximizing {
        if stand_pat >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat);
    } else {
        if stand_pat <= alpha {
            return alpha;
        }
        beta = beta.min(stand_pat);
    }
    
    // Only consider capture moves for quiescence
    let capture_moves = generate_capture_moves(game_state);
    
    if capture_moves.is_empty() {
        return stand_pat;
    }
    
    if maximizing {
        let mut max_eval = stand_pat;
        for m in capture_moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if !is_in_check(game_state, &Color::White) {  // Check if our king is safe
                let eval = quiescence_search(game_state, alpha, beta, false, depth - 1);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
            }
            undo_move(game_state, m, undo);
            if beta <= alpha {
                break;
            }
        }
        max_eval
    } else {
        let mut min_eval = stand_pat;
        for m in capture_moves.iter() {
            let undo = apply_move_in_place(game_state, m);
            if !is_in_check(game_state, &Color::Black) {  // Check if our king is safe
                let eval = quiescence_search(game_state, alpha, beta, true, depth - 1);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);
            }
            undo_move(game_state, m, undo);
            if beta <= alpha {
                break;
            }
        }
        min_eval
    }
}

#[allow(dead_code)]
fn generate_capture_moves(game_state: &GameState) -> Vec<ChessMove> {
    generate_legal_moves(game_state).into_iter()
        .filter(|m| game_state.board[m.to_row as usize][m.to_col as usize].is_some())
        .collect()
}

fn sort_moves(game_state: &GameState, moves: &[ChessMove]) -> Vec<ChessMove> {
    let mut scored_moves: Vec<(ChessMove, i32)> = moves.iter().map(|m| {
        let mut score = 0;
        
        // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        if let Some(target) = &game_state.board[m.to_row as usize][m.to_col as usize] {
            score += piece_value(&target.piece_type) * 10;
            
            if let Some(attacker) = &game_state.board[m.from_row as usize][m.from_col as usize] {
                score -= piece_value(&attacker.piece_type) / 10;
            }
        }
        
        // Bonus for promotions
        if m.promotion.is_some() {
            score += 900; // Queen promotion value
        }
        
        // Prefer center control for pawns in early game
        if let Some(piece) = &game_state.board[m.from_row as usize][m.from_col as usize] {
            if piece.piece_type == PieceType::Pawn {
                let center_dist = (3.5 - m.to_row as f32).abs() + (3.5 - m.to_col as f32).abs();
                score += (4.0 - center_dist) as i32 * 5;
            }
        }
        
        (m.clone(), score)
    }).collect();
    
    scored_moves.sort_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(m, _)| m).collect()
}

fn piece_value(piece_type: &PieceType) -> i32 {
    match piece_type {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 20000,
    }
}

// Generate legal moves without checking if they put our king in check - basic move generation
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
    moves
}

// Generate legal moves and filter those that would leave king in check
fn generate_legal_moves_with_check_validation(game_state: &mut GameState) -> Vec<ChessMove> {
    let moves = generate_legal_moves(game_state);
    let color = game_state.current_turn.clone();
    
    moves.into_iter().filter(|m| {
        let undo = apply_move_in_place(game_state, m);
        let valid = !is_in_check(game_state, &color);
        undo_move(game_state, m, undo);
        valid
    }).collect()
}

// Pawn move generation with corrected promotion and en passant
fn generate_pawn_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = match &game_state.board[row][col] {
        Some(p) => p,
        None => return moves,
    };
    
    let direction = if piece.color == Color::White { -1 } else { 1 };
    let start_row = if piece.color == Color::White { 6 } else { 1 };
    let promotion_row = if piece.color == Color::White { 0 } else { 7 };
    
    // Forward move
    let new_row = (row as i32 + direction) as i32;
    if new_row >= 0 && new_row < 8 && game_state.board[new_row as usize][col].is_none() {
        // Handle promotion
        if new_row as usize == promotion_row {
            for promo in &["queen", "rook", "bishop", "knight"] {
                moves.push(ChessMove {
                    from_row: row as i32,
                    from_col: col as i32,
                    to_row: new_row,
                    to_col: col as i32,
                    promotion: Some(promo.to_string()),
                });
            }
        } else {
            moves.push(ChessMove {
                from_row: row as i32,
                from_col: col as i32,
                to_row: new_row,
                to_col: col as i32,
                promotion: None,
            });
            
            // Double move from starting position
            if row == start_row {
                let double_row = (row as i32 + 2 * direction) as i32;
                if double_row >= 0 && double_row < 8 && game_state.board[double_row as usize][col].is_none() {
                    moves.push(ChessMove {
                        from_row: row as i32,
                        from_col: col as i32,
                        to_row: double_row,
                        to_col: col as i32,
                        promotion: None,
                    });
                }
            }
        }
    }
    
    // Captures
    for dc in &[-1, 1] {
        let new_col = col as i32 + dc;
        if new_col >= 0 && new_col < 8 {
            // Regular capture
            if let Some(target) = &game_state.board[new_row as usize][new_col as usize] {
                if target.color != piece.color {
                    if new_row as usize == promotion_row {
                        for promo in &["queen", "rook", "bishop", "knight"] {
                            moves.push(ChessMove {
                                from_row: row as i32,
                                from_col: col as i32,
                                to_row: new_row,
                                to_col: new_col,
                                promotion: Some(promo.to_string()),
                            });
                        }
                    } else {
                        moves.push(ChessMove {
                            from_row: row as i32,
                            from_col: col as i32,
                            to_row: new_row,
                            to_col: new_col,
                            promotion: None,
                        });
                    }
                }
            }
            
            // En passant capture
            if let Some((ep_row, ep_col)) = game_state.en_passant_target {
                if new_row == ep_row && new_col == ep_col {
                    moves.push(ChessMove {
                        from_row: row as i32,
                        from_col: col as i32,
                        to_row: new_row,
                        to_col: new_col,
                        promotion: None,
                    });
                }
            }
        }
    }
    
    moves
}

fn generate_knight_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = match &game_state.board[row][col] {
        Some(p) => p,
        None => return moves,
    };
    
    let offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
    for (dr, dc) in &offsets {
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
    moves
}

fn generate_bishop_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = match &game_state.board[row][col] {
        Some(p) => p,
        None => return moves,
    };
    
    for (dr, dc) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            let target = &game_state.board[r as usize][c as usize];
            if target.is_none() {
                moves.push(ChessMove {
                    from_row: row as i32,
                    from_col: col as i32,
                    to_row: r,
                    to_col: c,
                    promotion: None,
                });
            } else {
                if target.as_ref().unwrap().color != piece.color {
                    moves.push(ChessMove {
                        from_row: row as i32,
                        from_col: col as i32,
                        to_row: r,
                        to_col: c,
                        promotion: None,
                    });
                }
                break;
            }
            r += dr;
            c += dc;
        }
    }
    moves
}

fn generate_rook_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = match &game_state.board[row][col] {
        Some(p) => p,
        None => return moves,
    };
    
    for (dr, dc) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            let target = &game_state.board[r as usize][c as usize];
            if target.is_none() {
                moves.push(ChessMove {
                    from_row: row as i32,
                    from_col: col as i32,
                    to_row: r,
                    to_col: c,
                    promotion: None,
                });
            } else {
                if target.as_ref().unwrap().color != piece.color {
                    moves.push(ChessMove {
                        from_row: row as i32,
                        from_col: col as i32,
                        to_row: r,
                        to_col: c,
                        promotion: None,
                    });
                }
                break;
            }
            r += dr;
            c += dc;
        }
    }
    moves
}

fn generate_queen_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = generate_bishop_moves(game_state, row, col);
    moves.extend(generate_rook_moves(game_state, row, col));
    moves
}

fn generate_king_moves(game_state: &GameState, row: usize, col: usize) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    let piece = match &game_state.board[row][col] {
        Some(p) => p,
        None => return moves,
    };
    
    // Normal king moves
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 { continue; }
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
    }
    
    // Castling
    if piece.color == Color::White && row == 7 && col == 4 {
        // White kingside castling
        if game_state.castling_rights.white_kingside 
           && game_state.board[7][5].is_none() 
           && game_state.board[7][6].is_none() 
           && !is_square_attacked(game_state, 7, 4, &Color::Black)
           && !is_square_attacked(game_state, 7, 5, &Color::Black)
           && !is_square_attacked(game_state, 7, 6, &Color::Black) {
            moves.push(ChessMove {
                from_row: 7,
                from_col: 4,
                to_row: 7,
                to_col: 6,
                promotion: None,
            });
        }
        
        // White queenside castling
        if game_state.castling_rights.white_queenside 
           && game_state.board[7][3].is_none() 
           && game_state.board[7][2].is_none() 
           && game_state.board[7][1].is_none()
           && !is_square_attacked(game_state, 7, 4, &Color::Black)
           && !is_square_attacked(game_state, 7, 3, &Color::Black)
           && !is_square_attacked(game_state, 7, 2, &Color::Black) {
            moves.push(ChessMove {
                from_row: 7,
                from_col: 4,
                to_row: 7,
                to_col: 2,
                promotion: None,
            });
        }
    } else if piece.color == Color::Black && row == 0 && col == 4 {
        // Black kingside castling
        if game_state.castling_rights.black_kingside 
           && game_state.board[0][5].is_none() 
           && game_state.board[0][6].is_none()
           && !is_square_attacked(game_state, 0, 4, &Color::White)
           && !is_square_attacked(game_state, 0, 5, &Color::White)
           && !is_square_attacked(game_state, 0, 6, &Color::White) {
            moves.push(ChessMove {
                from_row: 0,
                from_col: 4,
                to_row: 0,
                to_col: 6,
                promotion: None,
            });
        }
        
        // Black queenside castling
        if game_state.castling_rights.black_queenside 
           && game_state.board[0][3].is_none() 
           && game_state.board[0][2].is_none() 
           && game_state.board[0][1].is_none()
           && !is_square_attacked(game_state, 0, 4, &Color::White)
           && !is_square_attacked(game_state, 0, 3, &Color::White)
           && !is_square_attacked(game_state, 0, 2, &Color::White) {
            moves.push(ChessMove {
                from_row: 0,
                from_col: 4,
                to_row: 0,
                to_col: 2,
                promotion: None,
            });
        }
    }
    
    moves
}

fn is_square_attacked(game_state: &GameState, row: usize, col: usize, by_color: &Color) -> bool {
    
    // Temporarily change turn to checking color
    let mut temp_state = game_state.clone();
    temp_state.current_turn = by_color.clone();
    
    // Generate all moves for the attacking color
    let moves = generate_legal_moves(&temp_state);
    
    // Check if any move attacks the square
    for m in moves {
        if m.to_row as usize == row && m.to_col as usize == col {
            return true;
        }
    }
    
    false
}

fn apply_move_in_place(game_state: &mut GameState, chess_move: &ChessMove) -> UndoInfo {
    let from_row = chess_move.from_row as usize;
    let from_col = chess_move.from_col as usize;
    let to_row = chess_move.to_row as usize;
    let to_col = chess_move.to_col as usize;

    let piece = game_state.board[from_row][from_col].as_ref().unwrap().clone();
    let captured = game_state.board[to_row][to_col].clone();
    let original_en_passant = game_state.en_passant_target.clone();
    let original_castling = game_state.castling_rights.clone();
    
    // Reset en passant target
    game_state.en_passant_target = None;
    
    // Handle pawn double move (set en passant target)
    if piece.piece_type == PieceType::Pawn && (to_row as i32 - from_row as i32).abs() == 2 {
        let ep_row = (from_row as i32 + to_row as i32) / 2;
        game_state.en_passant_target = Some((ep_row, from_col as i32));
    }

// Handle castling rights updates
if piece.piece_type == PieceType::King {
    if piece.color == Color::White {
        game_state.castling_rights.white_kingside = false;
        game_state.castling_rights.white_queenside = false;
    } else {
        game_state.castling_rights.black_kingside = false;
        game_state.castling_rights.black_queenside = false;
    }
} else if piece.piece_type == PieceType::Rook {
    if from_row == 7 && from_col == 0 && piece.color == Color::White {
        game_state.castling_rights.white_queenside = false;
    } else if from_row == 7 && from_col == 7 && piece.color == Color::White {
        game_state.castling_rights.white_kingside = false;
    } else if from_row == 0 && from_col == 0 && piece.color == Color::Black {
        game_state.castling_rights.black_queenside = false;
    } else if from_row == 0 && from_col == 7 && piece.color == Color::Black {
        game_state.castling_rights.black_kingside = false;
    }
}

// Check if capture removes castling rights (capturing opponent's rook)
if let Some(captured_piece) = &captured {
    if captured_piece.piece_type == PieceType::Rook {
        if to_row == 0 && to_col == 0 {
            game_state.castling_rights.black_queenside = false;
        } else if to_row == 0 && to_col == 7 {
            game_state.castling_rights.black_kingside = false;
        } else if to_row == 7 && to_col == 0 {
            game_state.castling_rights.white_queenside = false;
        } else if to_row == 7 && to_col == 7 {
            game_state.castling_rights.white_kingside = false;
        }
    }
}

// Handle castling move
let mut rook_from = None;
if piece.piece_type == PieceType::King && (from_col as i32 - to_col as i32).abs() == 2 {
    let (rook_from_col, rook_to_col) = if to_col > from_col {
        // Kingside
        (7, to_col - 1)
    } else {
        // Queenside
        (0, to_col + 1)
    };
    
    let rook_piece = game_state.board[from_row][rook_from_col].clone();
    rook_from = Some((from_row, rook_from_col, rook_piece.clone()));
    
    // Move the rook
    game_state.board[from_row][rook_to_col] = rook_piece;
    game_state.board[from_row][rook_from_col] = None;
}

// Handle en passant capture
if piece.piece_type == PieceType::Pawn && 
   original_en_passant.is_some() && 
   to_col as i32 == original_en_passant.unwrap().1 &&
   to_row as i32 == original_en_passant.unwrap().0 {
    
    let capture_row = from_row;
    game_state.board[capture_row][to_col] = None;
}

// Handle promotion
let promoted_piece = if let Some(promo_type) = &chess_move.promotion {
    let promotion_piece_type = match promo_type.as_str() {
        "queen" => PieceType::Queen,
        "rook" => PieceType::Rook,
        "bishop" => PieceType::Bishop,
        "knight" => PieceType::Knight,
        _ => piece.piece_type.clone(),
    };
    
    Some(Piece {
        piece_type: promotion_piece_type,
        color: piece.color.clone(),
    })
} else {
    None
};

// Move the piece
if let Some(promo) = promoted_piece {
    game_state.board[to_row][to_col] = Some(promo);
} else {
    game_state.board[to_row][to_col] = Some(piece.clone());
}
game_state.board[from_row][from_col] = None;

// Switch turn
game_state.current_turn = match game_state.current_turn {
    Color::White => Color::Black,
    Color::Black => Color::White,
};

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

// Restore the original piece to its starting position
game_state.board[from_row][from_col] = undo.from_square;

// Restore the captured piece (if any)
game_state.board[to_row][to_col] = undo.captured;

// Restore castling rook
if let Some((rook_row, rook_col, rook_piece)) = undo.rook_from {
    // Calculate where the rook was moved to
    let rook_to_col = if rook_col == 0 { to_col + 1 } else { to_col - 1 };
    
    // Restore the rook
    game_state.board[rook_row][rook_col] = rook_piece;
    game_state.board[rook_row][rook_to_col] = None;
}

// Restore en passant target
game_state.en_passant_target = undo.en_passant_target;

// Restore castling rights
game_state.castling_rights = undo.castling_rights;

// Switch turn back
game_state.current_turn = match game_state.current_turn {
    Color::White => Color::Black,
    Color::Black => Color::White,
};
}

fn evaluate_position(game_state: &GameState) -> i32 {
let mut score = 0;

// Piece values and basic position evaluation
for row in 0..8 {
    for col in 0..8 {
        if let Some(piece) = &game_state.board[row][col] {
            let piece_value = match piece.piece_type {
                PieceType::Pawn => 100,
                PieceType::Knight => 320,
                PieceType::Bishop => 330,
                PieceType::Rook => 500,
                PieceType::Queen => 900,
                PieceType::King => 20000,
            };
            
            // Basic positional bonuses
            let position_bonus = match piece.piece_type {
                PieceType::Pawn => {
                    // Encourage pawn advancement
                    let row_bonus = if piece.color == Color::White { 7 - row } else { row };
                    (row_bonus as i32) * 5
                },
                PieceType::Knight => {
                    // Knights are better in the center
                    let center_dist = (3.5 - row as f32).abs() + (3.5 - col as f32).abs();
                    (4.0 - center_dist) as i32 * 10
                },
                PieceType::Bishop => {
                    // Bishops are better on long diagonals
                    let center_dist = (3.5 - row as f32).abs() + (3.5 - col as f32).abs();
                    (4.0 - center_dist) as i32 * 5
                },
                PieceType::Rook => {
                    // Rooks are better on open files
                    let mut open_file = true;
                    for r in 0..8 {
                        if r != row && game_state.board[r][col].is_some() {
                            open_file = false;
                            break;
                        }
                    }
                    if open_file { 20 } else { 0 }
                },
                PieceType::Queen => {
                    // Queens are slightly better in the center
                    let center_dist = (3.5 - row as f32).abs() + (3.5 - col as f32).abs();
                    (4.0 - center_dist) as i32 * 3
                },
                PieceType::King => {
                    // Kings are safer at the edges in midgame/endgame
                    if is_endgame(game_state) {
                        // In endgame, king should move to center
                        let center_dist = (3.5 - row as f32).abs() + (3.5 - col as f32).abs();
                        (4.0 - center_dist) as i32 * 10
                    } else {
                        // In midgame, king safety is important
                        let edge_dist = (row as i32).min(7 - row as i32).min(col as i32).min(7 - col as i32);
                        edge_dist * 10
                    }
                },
            };
            
            // Add to score based on color
            let value = piece_value + position_bonus;
            if piece.color == Color::White {
                score += value;
            } else {
                score -= value;
            }
        }
    }
}

// Evaluate white mobility
let mut game_state_copy = game_state.clone();
game_state_copy.current_turn = Color::White;
let white_moves = generate_legal_moves_with_check_validation(&mut game_state_copy);
score += white_moves.len() as i32 * 5;

// Evaluate black mobility
game_state_copy.current_turn = Color::Black;
let black_moves = generate_legal_moves_with_check_validation(&mut game_state_copy);
score -= black_moves.len() as i32 * 5;

score
}

// Helper function to determine if we're in endgame (simplified)
fn is_endgame(game_state: &GameState) -> bool {
let mut queens = 0;
let mut minor_pieces = 0;

for row in 0..8 {
    for col in 0..8 {
        if let Some(piece) = &game_state.board[row][col] {
            match piece.piece_type {
                PieceType::Queen => queens += 1,
                PieceType::Bishop | PieceType::Knight => minor_pieces += 1,
                _ => {}
            }
        }
    }
}

queens == 0 || (queens == 2 && minor_pieces <= 2)
}

fn is_in_check(game_state: &mut GameState, color: &Color) -> bool {
// Find king position
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
    if king_pos.is_some() {
        break;
    }
}

if king_pos.is_none() {
    // Something is wrong if there's no king
    return false;
}

let (king_row, king_col) = king_pos.unwrap();

// Check if any opponent piece can attack the king
let opponent_color = match color {
    Color::White => Color::Black,
    Color::Black => Color::White,
};

// Generate opponent's moves (without check validation to avoid infinite recursion)
let original_turn = game_state.current_turn.clone();
game_state.current_turn = opponent_color.clone();
let opponent_moves = generate_legal_moves(game_state);
game_state.current_turn = original_turn;

// Check if any move attacks the king
for m in opponent_moves {
    if m.to_row as usize == king_row && m.to_col as usize == king_col {
        return true;
    }
}

false
}