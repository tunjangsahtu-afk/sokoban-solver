// Sokoban solver: push-optimal A* per room, written from scratch.
// Each level is a hallway with 1-4 rooms off it; a box pushed into the hallway
// can never return, so rooms are independent and the level optimum is the sum
// of the room optima. Every room is solved with A* over pushes: a state is the
// boxes plus the player's reachable region, the heuristic is the minimum-cost
// matching of boxes to goals over precomputed in-room distances (admissible,
// so the first solved state is push-optimal). Dead squares and frozen 2x2
// blocks are pruned. Work is bounded by a fixed state cap per room, never by
// the clock, so two runs answer byte-identically.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
const MAX_STATES = 250000;

function parse(text: string) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Squares a box can ever be pushed from onto a goal, found by pulling back
// from the goals: a box on square y can be pushed to x when the player can
// stand on the far side of y. A box off a live square is lost for good.
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const q = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < q.length; qi++) {
    const x = q[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
    }
  }
  return live;
}

// Split the board into rooms: cut the hallway row's dead squares out of the
// floor; each remaining connected part holding boxes is a room. Null when the
// board is no set of rooms (a box in the hallway, or a part whose boxes and
// goals do not match).
function roomsOf(lv: ReturnType<typeof parse>): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Flood fill from `from` over open squares with `occupied` in the way;
// marks the region with `stamp` in `seen` and returns its lowest square.
function reach(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    }
  }
  return low;
}

// A pushed box is frozen when it lands inside a 2x2 block of walls and boxes
// holding a box off its goal: nothing in that block can ever move again.
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` over open squares with `occupied` in the
// way, in lower LURD letters. Directions are tried u, d, l, r, so ties break
// the same way on every run.
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const p = q[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const s = p + dy * w + dx;
      if (wall[s] || occupied[s] || prev[s] !== -1) continue;
      prev[s] = p;
      how[s] = letter;
      q.push(s);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

function keyOf(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

// Push-optimal A* for one (sub-)level: boxes and goals are the room's (or the
// whole level's when rooms do not apply), `wall` already has every other
// room's boxes standing as walls, `area` is the squares boxes may travel on
// (the room's component, or every open square). Returns the moves string, or
// null when the state cap runs out or the level has no solution.
function solveAStar(
  w: number, wall: Uint8Array, goals: number[], boxes0: number[], player0: number,
  area: number[], maxStates: number,
): string | null {
  const size = wall.length;
  const n = boxes0.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  if (boxes0.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  // Distances from each goal over the area, for the matching heuristic.
  const areaSet = new Uint8Array(size);
  for (const s of area) areaSet[s] = 1;
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const s = p + dy * w + dx;
        if (areaSet[s] && d[s] < 0) { d[s] = d[p]! + 1; q.push(s); }
      }
    }
    return d;
  });
  // All assignments of goals to boxes, as index permutations.
  const perms: number[][] = [];
  const buildPerm = (used: boolean[], cur: number[]): void => {
    if (cur.length === n) { perms.push([...cur]); return; }
    for (let i = 0; i < n; i++) if (!used[i]) { used[i] = true; cur.push(i); buildPerm(used, cur); cur.pop(); used[i] = false; }
  };
  buildPerm(new Array(n).fill(false), []);
  const heuristic = (bs: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]!]![bs[i]!]!;
        if (d < 0) { sum = Infinity; break; }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    return best;
  };
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [player0];
  const sG: number[] = [0];
  const sKey: number[] = [];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const bestG = new Map<number, number>();
  for (const b of boxes0) occupied[b] = 1;
  const k0 = keyOf(boxes0, reach(w, wall, occupied, player0, seen, ++stamp, stack), size);
  sKey.push(k0);
  bestG.set(k0, 0);
  for (const b of boxes0) occupied[b] = 0;
  // Binary heap on (f, -g, seq): lowest f first, deeper first on ties, then
  // oldest first, so the order is fixed on every run.
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let seq = 0;
  const h0 = heuristic(boxes0);
  if (h0 === Infinity) return null;
  hpush([h0, 0, seq++, 0]);
  let found = solved(boxes0) ? 0 : -1;
  let expanded = 0;
  while (heap.length > 0 && found < 0) {
    const [, ng, , s] = hpop();
    const g = -ng;
    const bs = sBoxes[s]!;
    if (bestG.get(sKey[s]!) !== g) continue;
    if (expanded++ >= maxStates) return null;
    for (const b of bs) occupied[b] = 1;
    const here = ++stamp;
    reach(w, wall, occupied, sPlayer[s]!, seen, here, stack);
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi]!;
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let nk = -1;
        let next: number[] | null = null;
        let nh = Infinity;
        if (!frozen) {
          next = bs.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          nk = keyOf(next, reach(w, wall, occupied, b, mark, ++stamp, stack), size);
          nh = heuristic(next);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || nh === Infinity || next === null) continue;
        const seenG = bestG.get(nk);
        if (seenG !== undefined && seenG <= g + 1) continue;
        if (sBoxes.length >= maxStates) { for (const x of bs) occupied[x] = 0; return null; }
        bestG.set(nk, g + 1);
        const idx = sBoxes.length;
        sBoxes.push(next);
        sKey.push(nk);
        sPlayer.push(b);
        sG.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        pushDir.push(k);
        if (solved(next)) { found = idx; break; }
        hpush([g + 1 + nh, -(g + 1), seq++, idx]);
      }
    }
    for (const x of bs) occupied[x] = 0;
  }
  if (found < 0) return null;
  // Reconstruct: walk the player to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0.slice();
  for (const s of path) {
    const next = sBoxes[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[pushDir[s]!]![3];
    player = from;
    boxes = next;
  }
  return moves;
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solve(text: string): string {
  const lv = parse(text);
  const { w, wall, goals, boxes, player } = lv;
  const rooms = roomsOf(lv);
  if (rooms === null) {
    const area: number[] = [];
    for (let i = 0; i < wall.length; i++) if (!wall[i]) area.push(i);
    return solveAStar(w, wall, goals, boxes, player, area, MAX_STATES) ?? '';
  }
  let cur = player;
  let moves = '';
  const doneRooms = new Set<object>();
  for (const room of rooms) {
    const subWall = wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of doneRooms.has(other) ? other.goals : other.boxes) subWall[b] = 1;
    }
    const sub = solveAStar(w, subWall, room.goals, room.boxes, cur, room.squares, MAX_STATES);
    if (sub === null) return '';
    moves += sub;
    cur = endOf(w, cur, sub);
    doneRooms.add(room);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
