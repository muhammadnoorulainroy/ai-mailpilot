/**
 * Tests for the Pairing redemption flow, covering single-use token handoff,
 * wrong-code rejection, and brute-force lockout.
 */
import { describe, it, expect } from 'vitest';
import pino from 'pino';
import { Pairing } from '../src/pairing.js';

const silent = pino({ level: 'silent' });
/** Builds a Pairing seeded with a fixed token and the given pairing code. */
const make = (code = '123456'): Pairing => new Pairing('THE_TOKEN', silent, code);

describe('Pairing', () => {
  it('redeems the correct code once, returning the token', () => {
    const p = make();
    expect(p.redeem('123456')).toEqual({ token: 'THE_TOKEN' });
  });

  it('is single-use: a second redeem is rejected', () => {
    const p = make();
    p.redeem('123456');
    const again = p.redeem('123456');
    expect(again).toMatchObject({ status: 410 });
  });

  it('rejects a wrong code as unauthorized', () => {
    const p = make();
    expect(p.redeem('000000')).toMatchObject({ status: 401 });
  });

  it('burns the code after too many wrong attempts (no brute force)', () => {
    const p = make();
    for (let i = 0; i < 10; i++) expect(p.redeem('000000')).toMatchObject({ status: 401 });
    expect(p.redeem('123456')).toMatchObject({ status: 429 });
  });
});
