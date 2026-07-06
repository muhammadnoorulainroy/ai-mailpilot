/**
 * Fastify route handlers for account CRUD operations (list, get, create, delete).
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import type { Account } from '../repositories/account-repository.js';
import type { AccountDto } from '@ai-mailpilot/shared';

const CreateAccountBody = z.object({
  address: z.string().email(),
  displayName: z.string().optional(),
  kind: z.enum(['personal', 'work', 'institutional']),
  discoveryEnabled: z.boolean().optional(),
});

const UpdateDiscoveryBody = z.object({
  discoveryEnabled: z.boolean(),
});

/**
 * Maps an internal account record to the public DTO shape returned by the API.
 */
function toDto(account: Account): AccountDto {
  return {
    id: account.id,
    address: account.address,
    displayName: account.displayName,
    kind: account.kind,
    discoveryEnabled: account.discoveryEnabled,
    createdAt: account.createdAt,
  };
}

/**
 * Registers the account CRUD routes (list, get, create, delete) on the Fastify instance.
 */
export async function registerAccountRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.get('/accounts', async () => {
    const accounts = ctx.repos.accounts.list();
    return { accounts: accounts.map(toDto) };
  });

  app.get<{ Params: { id: string } }>('/accounts/:id', async (req, reply) => {
    const account = ctx.repos.accounts.findById(req.params.id);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    return { account: toDto(account) };
  });

  app.post('/accounts', async (req, reply) => {
    const parsed = CreateAccountBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.upsertByAddress(parsed.data);
    return { account: toDto(account) };
  });

  app.patch<{ Params: { id: string } }>('/accounts/:id/discovery', async (req, reply) => {
    const parsed = UpdateDiscoveryBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.setDiscoveryEnabled(
      req.params.id,
      parsed.data.discoveryEnabled,
    );
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    return { account: toDto(account) };
  });

  app.delete<{ Params: { id: string } }>('/accounts/:id', async (req, reply) => {
    const deleted = ctx.repos.accounts.delete(req.params.id);
    if (!deleted) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    reply.code(204).send();
  });
}
