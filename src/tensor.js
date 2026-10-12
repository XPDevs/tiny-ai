/**
 * ============================================================================
 * TinyAI - src/tensor.js
 * ============================================================================
 *
 * WHAT IS THIS FILE?
 * ------------------
 * This file contains TinyAI's Tensor class and its *automatic differentiation*
 * engine. Everything else in the framework (attention, transformers, the
 * optimizer, the trainer) is built on top of the handful of operations that
 * live here.
 *
 * If you have never built a neural network before, read this file first. It is
 * the mathematical foundation of the whole project.
 *
 *
 * THE BIG IDEA: A TENSOR IS JUST A LIST OF NUMBERS
 * ------------------------------------------------
 * A "tensor" is the generalization of a number, a vector and a matrix:
 *
 *   - scalar  : a single number              shape = []
 *   - vector  : a list of numbers            shape = [n]
 *   - matrix  : a grid of numbers            shape = [rows, cols]
 *   - tensor  : any number of dimensions     shape = [d0, d1, d2, ...]
 *
 * We store the numbers in one flat `Float64Array` and remember the `shape`
 * separately. For example the matrix
 *
 *       [[1, 2, 3],
 *        [4, 5, 6]]
 *
 * is stored as the flat array [1,2,3,4,5,6] with shape [2,3]. We convert
 * between "flat index" and "multi-dimensional coordinates" using *strides*
 * (see the `strides()` helper below). This is exactly how NumPy, PyTorch and
 * every other array library works under the hood.
 *
 *
 * THE SECOND BIG IDEA: GRADIENTS & BACKPROPAGATION
 * ------------------------------------------------
 * Training a neural network means adjusting its numbers (the *parameters*) so
 * that the network makes better predictions. To know *how* to adjust them we
 * need the gradient of a "loss" with respect to every parameter - that is,
 * "if I nudge this parameter up a tiny bit, does the loss go up or down, and
 * by how much?"
 *
 * We compute those gradients with the *chain rule* from calculus. Instead of
 * deriving every formula by hand (which would be thousands of lines and easy
 * to get wrong), we let the computer do it by building a graph:
 *
 *   1. When we compute an operation (say `c = a * b`) we record:
 *        - which tensors produced `c` (its parents `a` and `b`)
 *        - a little function that, given the gradient flowing into `c`,
 *          produces the gradient flowing back into `a` and `b`.
 *
 *   2. After the forward pass we call `loss.backward()`. That walks the graph
 *      in reverse topological order, calling each little function. This is
 *      called *reverse-mode automatic differentiation*, or simply
 *      *backpropagation*.
 *
 * Every derivative below is written out explicitly in the comments so that you
 * can check the mathematics yourself. We are deliberately optimising for
 * readability, not for raw speed.
 *
 *
 * USAGE
 * -----
 *   const { tensor, matmul, add, Tensor } = require('./tensor.js');
 *
 *   const a = tensor([[1, 2], [3, 4]], true);   // requires gradients
 *   const b = tensor([[5, 6], [7, 8]], true);
 *   const c = matmul(a, b);
 *   const loss = sum(c);
 *   loss.backward();
 *   console.log(a.grad);   // dc/da
 *
 * ============================================================================
 */

'use strict';

/* ==========================================================================
 * SECTION 1 - Shape arithmetic helpers
 * ========================================================================== */

/**
 * Number of elements described by a shape, i.e. the product of its dimensions.
 * prod([2,3]) === 6, prod([]) === 1 (a scalar has one element).
 */
function prod(shape) {
  let p = 1;
  for (let i = 0; i < shape.length; i++) p *= shape[i];
  return p;
}

/**
 * "Strides" tell us how far apart two neighbouring elements are in the flat
 * array. For shape [2,3] the strides are [3,1]:
 *
 *   element (row, col) lives at flat index row*3 + col
 *
 * The last dimension always has stride 1. This is the standard C/NumPy
 * row-major ("lexicographic") memory layout.
 */
function strides(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

/**
 * Compute the shape that results from broadcasting two shapes together.
 *
 * *Broadcasting* lets us combine tensors of different shapes by conceptually
 * "stretching" size-1 dimensions. For example:
 *
 *    [B, T, D]  +  [D]        -> [B, T, D]   (add a bias vector to every row)
 *    [B, T, D]  +  [T, D]     -> [B, T, D]   (add per-position information)
 *    [B, T, D]  +  [B, T, 1]  -> [B, T, D]   (multiply by a per-token scalar)
 *
 * The rule: dimensions are aligned from the right. Two dimensions are
 * compatible if they are equal or if one of them is 1.
 */
function broadcastShapes(a, b) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const av = a[a.length - n + i];
    const bv = b[b.length - n + i];
    const A = av === undefined ? 1 : av;
    const B = bv === undefined ? 1 : bv;
    if (A === B) out[i] = A;
    else if (A === 1) out[i] = B;
    else if (B === 1) out[i] = A;
    else throw new Error('Cannot broadcast shapes [' + a + '] and [' + b + ']');
  }
  return out;
}

/**
 * Convert a flat index into multi-dimensional coordinates (in place).
 * `coords` must already be allocated with `shape.length` slots.
 */
function decodeIndex(flat, shape, strd, coords) {
  let rem = flat;
  for (let d = 0; d < shape.length; d++) {
    const c = Math.floor(rem / strd[d]);
    rem -= c * strd[d];
    coords[d] = c;
  }
}

/**
 * The inverse of `unbroadcastPrepared`: take a gradient array shaped
 * `outShape` and sum it back down to `inShape`. This is the derivative of
 * broadcasting: when we broadcast a tensor we *reuse* its values, so when we
 * differentiate we must *sum* the incoming gradient over all the places the
 * value was reused.
 */
function unbroadcast(grad, outShape, inShape) {
  // Fast path: same shape, so the gradient passes straight through. This is
  // by far the most common case and avoids a full copy + index decode.
  if (outShape.length === inShape.length) {
    let same = true;
    for (let i = 0; i < outShape.length; i++) {
      if (outShape[i] !== inShape[i]) {
        same = false;
        break;
      }
    }
    if (same) return grad;
  }

  const outRank = outShape.length;
  const inRank = inShape.length;
  const outStr = strides(outShape);
  const inStr = strides(inShape);
  const result = new Float64Array(prod(inShape));
  const coords = new Array(outRank);
  for (let idx = 0; idx < grad.length; idx++) {
    decodeIndex(idx, outShape, outStr, coords);
    let off = 0;
    for (let d = 0; d < inRank; d++) {
      const dim = inShape[d];
      if (dim !== 1) off += coords[outRank - inRank + d] * inStr[d];
    }
    result[off] += grad[idx];
  }
  return result;
}

/** dst[i] += src[i], element by element. */
function addInto(dst, src) {
  for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

/** Flatten a (possibly nested) JavaScript array into a flat one. */
function flattenNested(arr, out) {
  if (Array.isArray(arr)) {
    for (let i = 0; i < arr.length; i++) flattenNested(arr[i], out);
  } else {
    out.push(arr);
  }
  return out;
}

/** Infer the shape of a (possibly nested) JavaScript array. */
function inferShape(arr) {
  const shape = [];
  let cur = arr;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    cur = cur[0];
  }
  return shape;
}

/* ==========================================================================
 * SECTION 2 - Global "no-grad" switch
 * ========================================================================== */

/**
 * During training we need to record the graph so we can back-propagate. During
 * *inference* (generating text) we do not - recording it would just waste time
 * and memory. `noGrad(fn)` runs `fn` with graph recording switched off.
 */
let gradEnabled = true;

function isGradEnabled() {
  return gradEnabled;
}

/** Run `fn()` with automatic-differentiation recording disabled. */
function noGrad(fn) {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/* ==========================================================================
 * SECTION 3 - The Tensor class
 * ========================================================================== */

class Tensor {
  /**
   * @param {Float64Array|Array} data  flat data, OR a nested JS array
   * @param {Array<number>} [shape]    dimensions (inferred if data is nested)
   * @param {boolean} [requiresGrad]   should we compute gradients for it?
   */
  constructor(data, shape, requiresGrad = false) {
    if (typeof data === 'number') {
      data = [data];
      shape = [];
    } else if (Array.isArray(data) && Array.isArray(data[0])) {
      const inferred = inferShape(data);
      const flat = [];
      flattenNested(data, flat);
      data = flat;
      shape = inferred;
    }

    if (!shape) shape = [data.length];
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);

    if (flat.length !== prod(shape)) {
      throw new Error(
        'Data length ' + flat.length + ' does not match shape [' + shape + ']'
      );
    }

    this.data = flat;
    this.shape = shape.slice();
    this.size = flat.length;
    this.requiresGrad = !!requiresGrad;
    this.grad = null;
    this._prev = new Set();       // parent tensors
    this._backward = () => {};    // closure that pushes gradient to parents
  }

  /* --- convenience accessors -------------------------------------------- */

  /** Read one element by coordinates: t.at(0, 3). */
  at(...coords) {
    if (coords.length !== this.shape.length) {
      throw new Error('at() expects ' + this.shape.length + ' coordinates');
    }
    const str = strides(this.shape);
    let off = 0;
    for (let d = 0; d < coords.length; d++) off += coords[d] * str[d];
    return this.data[off];
  }

  /** Return the single number of a scalar tensor. */
  item() {
    if (this.size !== 1) throw new Error('item() only works on scalar tensors');
    return this.data[0];
  }

  /** A deep copy (data copied, gradient and graph not carried over). */
  clone() {
    return new Tensor(this.data.slice(), this.shape.slice(), this.requiresGrad);
  }

  /** Set the gradient to zeros. */
  zeroGrad() {
    this.grad = null;
  }

  toString() {
    return 'Tensor(shape=[' + this.shape + '], size=' + this.size + ')';
  }

  /* --- the backward pass ------------------------------------------------ */

  /**
   * Reverse-mode automatic differentiation.
   *
   * Starting from `this` (usually the scalar loss) we visit every tensor in
   * the graph exactly once in reverse topological order and ask each one to
   * push its gradient onto its parents.
   *
   * @param {Float64Array} [seed] gradient arriving at this tensor. Defaults to
   *                               ones, which is correct because d(loss)/d(loss)
   *                               is 1 for every element when loss is scalar.
   */
  backward(seed) {
    // 1. Build a topological ordering of the graph (parents before children).
    const topo = [];
    const visited = new Set();
    const build = (t) => {
      if (visited.has(t)) return;
      visited.add(t);
      for (const parent of t._prev) build(parent);
      topo.push(t);
    };
    build(this);

    // 2. Zero every gradient so repeated backward() calls do not accumulate.
    for (const t of topo) {
      if (t !== this && t.requiresGrad) t.grad = new Float64Array(t.size);
    }

    // 3. Seed the output gradient.
    if (seed) this.grad = seed;
    else this.grad = new Float64Array(this.size).fill(1);

    // 4. Walk backwards, pushing gradients.
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}

/** Make sure `t` has a gradient buffer, creating a zero one if needed. */
function ensureGrad(t) {
  if (!t.grad) t.grad = new Float64Array(t.size);
  return t.grad;
}

/* ==========================================================================
 * SECTION 4 - Tensor factories
 * ========================================================================== */

/** Create a tensor from a flat or nested array. */
function tensor(data, requiresGrad = false, shape) {
  return new Tensor(data, shape, requiresGrad);
}

/** A tensor of zeros with the given shape. */
function zeros(shape, requiresGrad = false) {
  return new Tensor(new Float64Array(prod(shape)), shape, requiresGrad);
}

/** A tensor of ones with the given shape. */
function ones(shape, requiresGrad = false) {
  return new Tensor(new Float64Array(prod(shape)).fill(1), shape, requiresGrad);
}

/**
 * A tensor of normally-distributed random numbers (mean 0, standard deviation
 * `std`). This is how neural-network weights are initialized. We use the
 * Box-Muller transform to turn two uniform numbers into two Gaussian ones:
 *
 *   z0 = sqrt(-2 ln u1) * cos(2 pi u2)
 *   z1 = sqrt(-2 ln u1) * sin(2 pi u2)
 */
function randn(shape, requiresGrad = false, std = 1) {
  const n = prod(shape);
  const data = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u)) * std;
    data[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) data[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return new Tensor(data, shape, requiresGrad);
}

/** Build an integer tensor from a 2D array of token ids (used for input ids). */
function fromIds(ids) {
  const rows = ids.length;
  const cols = ids[0].length;
  const flat = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) flat[i * cols + j] = ids[i][j];
  }
  return new Tensor(flat, [rows, cols], false);
}

/* ==========================================================================
 * SECTION 5 - Element-wise operations (with broadcasting)
 * ========================================================================== */

/**
 * Generic engine for element-wise binary operations with broadcasting.
 *
 * @param {Tensor} a
 * @param {Tensor} b
 * @param {function(number,number):number} fwd     forward value
 * @param {function(number,number,number):number} bwdA  d(out)/d(a)
 * @param {function(number,number,number):number} bwdB  d(out)/d(b)
 *
 * The backward pass is computed on the *broadcasted* shape and then summed
 * back down to each input's real shape via `unbroadcast` - that summation is
 * exactly the derivative of broadcasting.
 */
function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function binaryOp(a, b, fwd, bwdA, bwdB) {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize = prod(outShape);
  const track = gradEnabled && (a.requiresGrad || b.requiresGrad);
  const out = new Tensor(new Float64Array(outSize), outShape, track);
  const aData = a.data;
  const bData = b.data;
  const oData = out.data;

  // ---- Fast path 1: both operands have exactly the same shape. -----------
  // This is extremely common (residual adds, x*x, multiplying by a mask) and
  // needs no index arithmetic at all.
  if (shapesEqual(a.shape, b.shape)) {
    for (let i = 0; i < outSize; i++) oData[i] = fwd(aData[i], bData[i]);
    if (track) {
      out._prev.add(a);
      out._prev.add(b);
      out._backward = () => {
        const g = out.grad;
        if (a.requiresGrad) {
          const ga = ensureGrad(a);
          for (let i = 0; i < outSize; i++) ga[i] += bwdA(g[i], aData[i], bData[i]);
        }
        if (b.requiresGrad) {
          const gb = ensureGrad(b);
          for (let i = 0; i < outSize; i++) gb[i] += bwdB(g[i], aData[i], bData[i]);
        }
      };
    }
    return out;
  }

  // ---- Fast path 2: `b` is a scalar (rank 0, one element). ---------------
  if (b.size === 1 && b.shape.length === 0) {
    const bv = bData[0];
    for (let i = 0; i < outSize; i++) oData[i] = fwd(aData[i], bv);
    if (track) {
      out._prev.add(a);
      if (b.requiresGrad) out._prev.add(b);
      out._backward = () => {
        const g = out.grad;
        if (a.requiresGrad) {
          const ga = ensureGrad(a);
          for (let i = 0; i < outSize; i++) ga[i] += bwdA(g[i], aData[i], bv);
        }
        if (b.requiresGrad) {
          let s = 0;
          for (let i = 0; i < outSize; i++) s += bwdB(g[i], aData[i], bv);
          ensureGrad(b)[0] += s;
        }
      };
    }
    return out;
  }

  // ---- Fast path 3: `a` is a scalar. -------------------------------------
  if (a.size === 1 && a.shape.length === 0) {
    const av = aData[0];
    for (let i = 0; i < outSize; i++) oData[i] = fwd(av, bData[i]);
    if (track) {
      if (a.requiresGrad) out._prev.add(a);
      out._prev.add(b);
      out._backward = () => {
        const g = out.grad;
        if (a.requiresGrad) {
          let s = 0;
          for (let i = 0; i < outSize; i++) s += bwdA(g[i], av, bData[i]);
          ensureGrad(a)[0] += s;
        }
        if (b.requiresGrad) {
          const gb = ensureGrad(b);
          for (let i = 0; i < outSize; i++) gb[i] += bwdB(g[i], av, bData[i]);
        }
      };
    }
    return out;
  }

  // ---- General case: broadcasting with full index arithmetic. -------------
  const aStr = strides(a.shape);
  const bStr = strides(b.shape);
  const oStr = strides(outShape);
  const aRank = a.shape.length;
  const bRank = b.shape.length;
  const oRank = outShape.length;
  const coords = new Array(oRank);

  for (let idx = 0; idx < outSize; idx++) {
    decodeIndex(idx, outShape, oStr, coords);
    let aOff = 0;
    let bOff = 0;
    for (let d = 0; d < aRank; d++) {
      if (a.shape[d] !== 1) aOff += coords[oRank - aRank + d] * aStr[d];
    }
    for (let d = 0; d < bRank; d++) {
      if (b.shape[d] !== 1) bOff += coords[oRank - bRank + d] * bStr[d];
    }
    oData[idx] = fwd(aData[aOff], bData[bOff]);
  }

  if (track) {
    out._prev.add(a);
    out._prev.add(b);
    out._backward = () => {
      const gA = a.requiresGrad ? new Float64Array(outSize) : null;
      const gB = b.requiresGrad ? new Float64Array(outSize) : null;
      for (let idx = 0; idx < outSize; idx++) {
        decodeIndex(idx, outShape, oStr, coords);
        let aOff = 0;
        let bOff = 0;
        for (let d = 0; d < aRank; d++) {
          if (a.shape[d] !== 1) aOff += coords[oRank - aRank + d] * aStr[d];
        }
        for (let d = 0; d < bRank; d++) {
          if (b.shape[d] !== 1) bOff += coords[oRank - bRank + d] * bStr[d];
        }
        const g = out.grad[idx];
        if (gA) gA[idx] = bwdA(g, aData[aOff], bData[bOff]);
        if (gB) gB[idx] = bwdB(g, aData[aOff], bData[bOff]);
      }
      if (gA) addInto(a.grad, unbroadcast(gA, outShape, a.shape));
      if (gB) addInto(b.grad, unbroadcast(gB, outShape, b.shape));
    };
  }
  return out;
}

/** Wrap a plain number as a scalar tensor. */
function scalar(v) {
  return new Tensor(Float64Array.of(v), [], false);
}

/** a + b. Derivative: d/da = 1, d/db = 1. */
function add(a, b) {
  if (typeof a === 'number') a = scalar(a);
  if (typeof b === 'number') b = scalar(b);
  return binaryOp(a, b, (x, y) => x + y, (g) => g, (g) => g);
}

/** a - b. Derivative: d/da = 1, d/db = -1. */
function sub(a, b) {
  if (typeof a === 'number') a = scalar(a);
  if (typeof b === 'number') b = scalar(b);
  return binaryOp(a, b, (x, y) => x - y, (g) => g, (g) => -g);
}

/** a * b (element-wise). Derivative: d/da = b, d/db = a. */
function mul(a, b) {
  if (typeof a === 'number') a = scalar(a);
  if (typeof b === 'number') b = scalar(b);
  return binaryOp(a, b, (x, y) => x * y, (g, x, y) => g * y, (g, x, y) => g * x);
}

/** a / b (element-wise). Derivative: d/da = 1/b, d/db = -a / b^2. */
function div(a, b) {
  if (typeof a === 'number') a = scalar(a);
  if (typeof b === 'number') b = scalar(b);
  return binaryOp(
    a,
    b,
    (x, y) => x / y,
    (g, x, y) => g / y,
    (g, x, y) => (-g * x) / (y * y)
  );
}

/* ==========================================================================
 * SECTION 6 - Unary operations
 * ========================================================================== */

/**
 * Generic engine for element-wise unary operations.
 * @param {function(number):number} fwd
 * @param {function(number,number):number} bwd  (inputValue, incomingGrad) -> grad
 */
function unaryOp(a, fwd, bwd) {
  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(new Float64Array(a.size), a.shape.slice(), track);
  for (let i = 0; i < a.size; i++) out.data[i] = fwd(a.data[i]);
  if (track) {
    out._prev.add(a);
    out._backward = () => {
      const g = ensureGrad(a);
      for (let i = 0; i < a.size; i++) {
        g[i] += bwd(a.data[i], out.grad[i]);
      }
    };
  }
  return out;
}

/** e^x. Derivative: d/dx e^x = e^x. */
function exp(a) {
  return unaryOp(a, Math.exp, (x, g) => Math.exp(x) * g);
}

/** sqrt(x). Derivative: d/dx sqrt(x) = 1 / (2 sqrt(x)). */
function sqrt(a) {
  return unaryOp(a, Math.sqrt, (x, g) => (0.5 / Math.sqrt(x)) * g);
}

/** x^p for a constant p. Derivative: p * x^(p-1). */
function pow(a, p) {
  return unaryOp(a, (x) => Math.pow(x, p), (x, g) => p * Math.pow(x, p - 1) * g);
}

/** tanh(x). Derivative: 1 - tanh(x)^2. */
function tanh(a) {
  return unaryOp(a, Math.tanh, (x, g) => (1 - Math.tanh(x) * Math.tanh(x)) * g);
}

/**
 * GELU activation (the smooth one used by GPT models):
 *
 *   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
 *
 * We build it from the primitives above so that autograd derives the gradient
 * for us - you can read the formula directly in the code.
 */
function gelu(a) {
  const c = Math.sqrt(2 / Math.PI);
  const x3 = mul(pow(a, 3), 0.044715);
  const inner = mul(add(a, x3), c);
  const t = tanh(inner);
  return mul(mul(a, 0.5), add(t, 1));
}

/* ==========================================================================
 * SECTION 7 - Matrix multiplication
 * ========================================================================== */

/**
 * Matrix multiplication, generalized to batches of matrices.
 *
 * For 2D inputs this is the familiar C[i,j] = sum_k A[i,k] * B[k,j].
 * For higher rank inputs the leading dimensions are treated as a batch and
 * broadcast against each other. For example:
 *
 *   A: [B, T, K]   B: [K, N]      -> [B, T, N]
 *   A: [B, H, T, K] B: [B, H, K, T] -> [B, H, T, T]
 *
 * Backward (for C = A @ B):
 *   dA = dC @ B^T
 *   dB = A^T @ dC
 */
function matmul(a, b) {
  const aShape = a.shape;
  const bShape = b.shape;
  if (aShape.length < 2 || bShape.length < 2) {
    throw new Error('matmul expects tensors with rank >= 2');
  }

  const M = aShape[aShape.length - 2];
  const K = aShape[aShape.length - 1];
  const K2 = bShape[bShape.length - 2];
  const N = bShape[bShape.length - 1];
  if (K !== K2) {
    throw new Error('matmul inner dimensions disagree: ' + K + ' vs ' + K2);
  }

  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch, bBatch);
  const batchCount = prod(batchShape);

  const aBatchStr = strides(aBatch);
  const bBatchStr = strides(bBatch);
  const batchStr = strides(batchShape);

  const track = gradEnabled && (a.requiresGrad || b.requiresGrad);
  const out = new Tensor(
    new Float64Array(batchCount * M * N),
    batchShape.concat([M, N]),
    track
  );

  // Precompute, for every batch, the offset into A and B.
  const aOffs = new Int32Array(batchCount);
  const bOffs = new Int32Array(batchCount);
  const coords = new Array(batchShape.length);
  for (let idx = 0; idx < batchCount; idx++) {
    decodeIndex(idx, batchShape, batchStr, coords);
    let ao = 0;
    for (let d = 0; d < aBatch.length; d++) {
      if (aBatch[d] !== 1) ao += coords[batchShape.length - aBatch.length + d] * aBatchStr[d];
    }
    let bo = 0;
    for (let d = 0; d < bBatch.length; d++) {
      if (bBatch[d] !== 1) bo += coords[batchShape.length - bBatch.length + d] * bBatchStr[d];
    }
    // `ao`/`bo` are offsets measured in *matrices*; scale them by the size of
    // one matrix to get the offset into the flat data array.
    aOffs[idx] = ao * M * K;
    bOffs[idx] = bo * K * N;
  }

  // Forward.
  for (let idx = 0; idx < batchCount; idx++) {
    const ao = aOffs[idx];
    const bo = bOffs[idx];
    const cOff = idx * M * N;
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const av = a.data[ao + i * K + k];
        if (av === 0) continue;
        const bRow = bo + k * N;
        const cRow = cOff + i * N;
        for (let j = 0; j < N; j++) out.data[cRow + j] += av * b.data[bRow + j];
      }
    }
  }

  // Backward.
  if (track) {
    out._prev.add(a);
    out._prev.add(b);
    out._backward = () => {
      const dA = a.requiresGrad ? new Float64Array(batchCount * M * K) : null;
      const dB = b.requiresGrad ? new Float64Array(batchCount * K * N) : null;
      for (let idx = 0; idx < batchCount; idx++) {
        const ao = aOffs[idx];
        const bo = bOffs[idx];
        const cOff = idx * M * N;
        if (dA) {
          const base = idx * M * K;
          for (let i = 0; i < M; i++) {
            for (let k = 0; k < K; k++) {
              let s = 0;
              for (let j = 0; j < N; j++) s += out.grad[cOff + i * N + j] * b.data[bo + k * N + j];
              dA[base + i * K + k] = s;
            }
          }
        }
        if (dB) {
          const base = idx * K * N;
          for (let k = 0; k < K; k++) {
            for (let j = 0; j < N; j++) {
              let s = 0;
              for (let i = 0; i < M; i++) s += a.data[ao + i * K + k] * out.grad[cOff + i * N + j];
              dB[base + k * N + j] = s;
            }
          }
        }
      }
      if (dA) addInto(a.grad, unbroadcast(dA, batchShape.concat([M, K]), aShape));
      if (dB) addInto(b.grad, unbroadcast(dB, batchShape.concat([K, N]), bShape));
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 8 - Reshape, transpose and slicing
 * ========================================================================== */

/**
 * Change the shape of a tensor without changing the order of its elements.
 * The total number of elements must stay the same. The gradient is simply the
 * incoming gradient reshaped back to the original shape.
 */
function reshape(a, shape) {
  if (prod(shape) !== a.size) {
    throw new Error('reshape: cannot reshape [' + a.shape + '] to [' + shape + ']');
  }
  const track = gradEnabled && a.requiresGrad;
  // We can safely share the data buffer because no operation mutates data
  // in place - every operation allocates a fresh output array.
  const out = new Tensor(a.data, shape, track);
  if (track) {
    out._prev.add(a);
    out._backward = () => {
      addInto(a.grad, out.grad);
    };
  }
  return out;
}

/**
 * Swap two axes of a tensor. `transpose(x)` (no arguments) swaps the last two
 * axes, turning [.., M, N] into [.., N, M].
 *
 * The permutation "swap d1 and d2" is its own inverse, so the backward pass is
 * exactly the same swap applied to the incoming gradient.
 */
function transpose(a, d1, d2) {
  const rank = a.shape.length;
  if (d1 === undefined) d1 = rank - 2;
  if (d2 === undefined) d2 = rank - 1;
  if (d1 < 0) d1 += rank;
  if (d2 < 0) d2 += rank;
  if (d1 < 0 || d2 < 0 || d1 >= rank || d2 >= rank) {
    throw new Error('transpose: axis out of range');
  }

  const outShape = a.shape.slice();
  const tmp = outShape[d1];
  outShape[d1] = outShape[d2];
  outShape[d2] = tmp;

  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(new Float64Array(a.size), outShape, track);
  const inStr = strides(a.shape);
  const outStr = strides(outShape);
  const coords = new Array(rank);

  for (let idx = 0; idx < a.size; idx++) {
    decodeIndex(idx, outShape, outStr, coords);
    let off = 0;
    for (let d = 0; d < rank; d++) {
      const c = d === d1 ? coords[d2] : d === d2 ? coords[d1] : coords[d];
      off += c * inStr[d];
    }
    out.data[idx] = a.data[off];
  }

  if (track) {
    out._prev.add(a);
    out._backward = () => {
      addInto(a.grad, transposeArray(out.grad, outShape, a.shape, d1, d2));
    };
  }
  return out;
}

/** Helper used by transpose().backward: apply the swap to a gradient buffer. */
function transposeArray(flat, fromShape, toShape, d1, d2) {
  const rank = fromShape.length;
  const fromStr = strides(fromShape);
  const toStr = strides(toShape);
  const res = new Float64Array(prod(toShape));
  const coords = new Array(rank);
  for (let idx = 0; idx < flat.length; idx++) {
    decodeIndex(idx, fromShape, fromStr, coords);
    let off = 0;
    for (let d = 0; d < rank; d++) {
      const c = d === d1 ? coords[d2] : d === d2 ? coords[d1] : coords[d];
      off += c * toStr[d];
    }
    res[off] += flat[idx];
  }
  return res;
}

/** Return the first `end - start` rows of a 2D tensor (used for positions). */
function sliceRows(a, start, end) {
  const cols = a.shape[1];
  const count = end - start;
  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(new Float64Array(count * cols), [count, cols], track);
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < cols; j++) out.data[i * cols + j] = a.data[(start + i) * cols + j];
  }
  if (track) {
    out._prev.add(a);
    out._backward = () => {
      const g = ensureGrad(a);
      for (let i = 0; i < count; i++) {
        for (let j = 0; j < cols; j++) g[(start + i) * cols + j] += out.grad[i * cols + j];
      }
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 9 - Reductions
 * ========================================================================== */

/**
 * Mean over the last dimension, keeping that dimension as size 1.
 * [.., N] -> [.., 1].
 *
 * Forward:  out = (1/N) * sum_j x_j
 * Backward: every element receives grad / N (because each element contributed
 *           equally to the mean).
 */
function meanLast(a) {
  const rank = a.shape.length;
  const last = a.shape[rank - 1];
  const rows = a.size / last;
  const outShape = a.shape.slice();
  outShape[rank - 1] = 1;
  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(new Float64Array(rows), outShape, track);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const base = r * last;
    for (let j = 0; j < last; j++) s += a.data[base + j];
    out.data[r] = s / last;
  }
  if (track) {
    out._prev.add(a);
    out._backward = () => {
      const g = ensureGrad(a);
      for (let r = 0; r < rows; r++) {
        const base = r * last;
        const gg = out.grad[r] / last;
        for (let j = 0; j < last; j++) g[base + j] += gg;
      }
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 10 - Softmax
 * ========================================================================== */

/**
 * Softmax over the last dimension:
 *
 *   softmax(x)_j = exp(x_j - max) / sum_k exp(x_k - max)
 *
 * Subtracting the maximum is a numerical-stability trick: exp(x) can overflow
 * for large x, but exp(x - max) is always <= 1. It does not change the result.
 *
 * `mask` (optional) is a flat 0/1 array of shape [T, T] where a 1 means
 * "forbid this position" (set its probability to zero). This is how causal
 * (a.k.a. autoregressive) attention stops a token from looking into the
 * future.
 *
 * Backward (for a single row, where s = softmax output and go = incoming grad):
 *   dx_j = s_j * (go_j - sum_k go_k * s_k)
 */
function softmax(a, mask) {
  const rank = a.shape.length;
  const last = a.shape[rank - 1];
  const rows = a.size / last;
  const T = rank >= 2 ? a.shape[rank - 2] : 1;
  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(new Float64Array(a.size), a.shape.slice(), track);

  for (let r = 0; r < rows; r++) {
    const base = r * last;
    const rowIn2 = r % T;
    let m = -Infinity;
    for (let j = 0; j < last; j++) {
      if (mask && mask[rowIn2 * last + j]) continue;
      const v = a.data[base + j];
      if (v > m) m = v;
    }
    if (m === -Infinity) m = 0;
    let sum = 0;
    for (let j = 0; j < last; j++) {
      if (mask && mask[rowIn2 * last + j]) {
        out.data[base + j] = 0;
        continue;
      }
      const e = Math.exp(a.data[base + j] - m);
      out.data[base + j] = e;
      sum += e;
    }
    if (sum === 0) sum = 1;
    for (let j = 0; j < last; j++) out.data[base + j] /= sum;
  }

  if (track) {
    out._prev.add(a);
    out._backward = () => {
      const g = ensureGrad(a);
      for (let r = 0; r < rows; r++) {
        const base = r * last;
        const rowIn2 = r % T;
        let dot = 0;
        for (let j = 0; j < last; j++) {
          if (mask && mask[rowIn2 * last + j]) continue;
          dot += out.grad[base + j] * out.data[base + j];
        }
        for (let j = 0; j < last; j++) {
          if (mask && mask[rowIn2 * last + j]) continue;
          g[base + j] += out.data[base + j] * (out.grad[base + j] - dot);
        }
      }
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 11 - Embedding lookup
 * ========================================================================== */

/**
 * Look up rows of an embedding matrix.
 *
 *   weight: [V, D]     (V = vocabulary size, D = embedding dimension)
 *   ids:    [a, b, ..] (any shape of integer token ids)
 *   out:    [a, b, .., D]
 *
 * The forward pass copies the row `weight[id]` for every id. The backward pass
 * scatters the incoming gradient back into the rows that were used - rows used
 * more than once simply accumulate more gradient.
 */
function embedding(weight, ids) {
  const D = weight.shape[weight.shape.length - 1];
  const V = weight.shape[0];
  const n = ids.size;
  const outShape = ids.shape.concat([D]);
  const track = gradEnabled && weight.requiresGrad;
  const out = new Tensor(new Float64Array(n * D), outShape, track);
  const idx = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    let id = Math.round(ids.data[i]);
    if (id < 0) id = 0;
    if (id >= V) id = V - 1;
    idx[i] = id;
    const src = id * D;
    const dst = i * D;
    for (let d = 0; d < D; d++) out.data[dst + d] = weight.data[src + d];
  }

  if (track) {
    out._prev.add(weight);
    out._backward = () => {
      const g = ensureGrad(weight);
      for (let i = 0; i < n; i++) {
        const src = i * D;
        const dst = idx[i] * D;
        for (let d = 0; d < D; d++) g[dst + d] += out.grad[src + d];
      }
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 12 - Cross-entropy loss
 * ========================================================================== */

/**
 * Cross-entropy loss for next-token prediction.
 *
 *   logits:  [N, V]   raw scores for N independent examples
 *   targets: [N]      the correct token id for each example
 *   loss:    scalar   the average negative log-likelihood
 *
 * For one example with logits z and correct class t:
 *
 *   p = softmax(z)
 *   loss = -log(p_t) = -(z_t - max - log(sum_j exp(z_j - max)))
 *
 * We implement it as a single fused operation (instead of composing softmax +
 * log + gather) because the softmax probabilities are needed for the gradient
 * anyway, and building them once is far cheaper.
 *
 * Backward: d(loss)/d(z_j) = (p_j - 1[j == t]) / N
 */
function crossEntropy(logits, targets) {
  const V = logits.shape[1];
  const N = logits.shape[0];
  const track = gradEnabled && logits.requiresGrad;
  const out = new Tensor(new Float64Array(1), [], track);
  const probs = track ? new Float64Array(N * V) : null;
  let loss = 0;

  for (let i = 0; i < N; i++) {
    const base = i * V;
    let m = -Infinity;
    for (let j = 0; j < V; j++) {
      const v = logits.data[base + j];
      if (v > m) m = v;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(logits.data[base + j] - m);
      if (probs) probs[base + j] = e;
      sum += e;
    }
    const t = Math.round(targets.data[i]);
    loss += -(logits.data[base + t] - m - Math.log(sum));
    if (probs) {
      for (let j = 0; j < V; j++) probs[base + j] /= sum;
    }
  }
  out.data[0] = loss / N;

  if (track) {
    out._prev.add(logits);
    out._backward = () => {
      const g = ensureGrad(logits);
      const scale = out.grad[0] / N;
      for (let i = 0; i < N; i++) {
        const t = Math.round(targets.data[i]);
        const base = i * V;
        for (let j = 0; j < V; j++) {
          g[base + j] += (probs[base + j] - (j === t ? 1 : 0)) * scale;
        }
      }
    };
  }
  return out;
}

/* ==========================================================================
 * SECTION 13 - Small utilities used for generation / reporting
 * ========================================================================== */

/** Index of the largest value in a plain array. */
function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/** Sum of all elements (returns a scalar tensor, differentiable). */
function sumAll(a) {
  const track = gradEnabled && a.requiresGrad;
  const out = new Tensor(Float64Array.of(0), [], track);
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  out.data[0] = s;
  if (track) {
    out._prev.add(a);
    out._backward = () => {
      const g = ensureGrad(a);
      const gg = out.grad[0];
      for (let i = 0; i < a.size; i++) g[i] += gg;
    };
  }
  return out;
}

/** Mean of every element (differentiable scalar). */
function meanAll(a) {
  return div(sumAll(a), a.size);
}

/* ==========================================================================
 * SECTION 14 - Exports
 * ========================================================================== */

const TinyTensor = {
  Tensor,
  tensor,
  zeros,
  ones,
  randn,
  scalar,
  fromIds,
  add,
  sub,
  mul,
  div,
  exp,
  sqrt,
  pow,
  tanh,
  gelu,
  matmul,
  reshape,
  transpose,
  sliceRows,
  meanLast,
  softmax,
  embedding,
  crossEntropy,
  sumAll,
  meanAll,
  argmax,
  noGrad,
  isGradEnabled,
  // low-level helpers, exposed for tests and for other modules
  prod,
  strides,
  broadcastShapes,
  unbroadcast,
  ensureGrad,
  addInto,
};

// Works both under Node (require) and in the browser (as a script tag).
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyTensor;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyTensor);
}
