export interface Email {
  messageId: string;
  subject: string;
  from: string;
  to: string[];
  cc: string[];
  date: string;
  body: string;
  folder: string;
  tags: string[];
  read: boolean;
  hasAttachments: boolean;
}

export interface EmailSummary {
  messageId: string;
  subject: string;
  from: string;
  date: string;
  folder: string;
  tags: string[];
  preview: string;
}
