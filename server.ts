import crypto from 'crypto';
import dotenv from 'dotenv';
import express, { type NextFunction, type Request, type Response } from 'express';
import multer from 'multer';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import {
  analyzeDocumentWithGemini,
  chatWithDocuments,
  detectDuplicatesWithGemini,
  extractTextFromPdfBuffer,
  findPotentialDocumentDuplicate,
  findPotentialTaskDuplicate,
  translateDocumentWithGemini,
  translateTasksWithGemini,
} from './server/gemini.ts';
import {
  appendChatMessages,
  clearAllState,
  computeContentHash,
  readState,
  resetToSampleState,
  writeState,
} from './server/store.ts';
import type {
  CampusDocument,
  CampusTask,
  ChatMessage,
  PriorityLevel,
  SummaryLanguage,
  TargetRole,
  TaskStatus,
} from './src/types.ts';

dotenv.config();

const PORT = Number(process.env.PORT) || 3000;
const ALLOWED_EXTENSIONS = new Set(['.pdf', '.jpg', '.jpeg', '.png', '.txt']);
const ALLOWED_MIME_TYPES = new Set([
  'application/pdf',
  'image/jpeg',
  'image/png',
  'text/plain',
]);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024, // 15 MB
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext) && !ALLOWED_MIME_TYPES.has(file.mimetype)) {
      cb(new Error('Unsupported file type. Upload PDF, JPG, PNG, or TXT documents only.'));
      return;
    }
    cb(null, true);
  },
});

function escapeCsvCell(value: string | null | undefined): string {
  let str = String(value ?? '');
  // Neutralize spreadsheet formula injection
  if (/^[=+\-@\t\r]/.test(str)) {
    str = `'${str}`;
  }
  const escaped = str.replace(/"/g, '""');
  return `"${escaped}"`;
}

const PRIORITY_RANK: Record<PriorityLevel, number> = {
  high: 3,
  medium: 2,
  low: 1,
};

const STATUS_RANK: Record<TaskStatus, number> = {
  completed: 3,
  in_progress: 2,
  pending: 1,
};

function clearAllDuplicateFlags(state: { documents: CampusDocument[]; tasks: CampusTask[] }) {
  for (const doc of state.documents) {
    doc.duplicateOfDocumentId = null;
    doc.duplicateScore = null;
    doc.duplicateReason = null;
    doc.duplicateMatchingFields = undefined;
  }
  for (const task of state.tasks) {
    task.duplicateOfTaskId = null;
    task.duplicateNote = null;
    task.duplicateScore = null;
    task.duplicateMatchingFields = undefined;
  }
}

function applyScanResultsToState(
  state: { documents: CampusDocument[]; tasks: CampusTask[] },
  scan: Awaited<ReturnType<typeof detectDuplicatesWithGemini>>
) {
  for (const dup of scan.documentDuplicates) {
    const target = state.documents.find((d) => d.id === dup.documentId);
    if (target) {
      target.duplicateOfDocumentId = dup.duplicateOfDocumentId;
      target.duplicateScore = dup.score;
      target.duplicateReason = dup.reason;
      target.duplicateMatchingFields = dup.matchingFields;
    }
  }
  for (const tDup of scan.taskDuplicates) {
    const target = state.tasks.find((t) => t.id === tDup.taskId);
    if (target) {
      target.duplicateOfTaskId = tDup.duplicateOfTaskId;
      target.duplicateNote = tDup.note;
      target.duplicateScore = tDup.score;
      target.duplicateMatchingFields = tDup.matchingFields;
    }
  }
}

function cloneExactDuplicateForReview(params: {
  state: ReturnType<typeof readState>;
  existingDoc: CampusDocument;
  fileName: string;
  fileType: CampusDocument['fileType'];
  mimeType: string;
  fileSize: number;
  extractionMethod: CampusDocument['extractionMethod'];
}): { document: CampusDocument; createdTasks: CampusTask[] } {
  const now = new Date().toISOString();
  const docId = `doc-${crypto.randomUUID().slice(0, 8)}`;

  const clonedDoc: CampusDocument = {
    id: docId,
    fileName: params.fileName,
    fileType: params.fileType,
    mimeType: params.mimeType,
    fileSize: params.fileSize,
    uploadedAt: now,
    extractionMethod: params.extractionMethod,
    rawText: params.existingDoc.rawText,
    contentHash: params.existingDoc.contentHash,
    facts: structuredClone(params.existingDoc.facts),
    aiRecommendations: structuredClone(params.existingDoc.aiRecommendations),
    translations: structuredClone(params.existingDoc.translations),
    duplicateOfDocumentId: params.existingDoc.id,
    duplicateScore: 100,
    duplicateReason: `Exact content match with "${params.existingDoc.facts.title}" (${params.existingDoc.fileName}).`,
    duplicateMatchingFields: ['Exact Content Hash', 'Notice Body'],
  };

  const existingTasks = params.state.tasks.filter((t) => t.documentId === params.existingDoc.id);
  const clonedTasks: CampusTask[] = existingTasks.map((t) => ({
    id: `task-${crypto.randomUUID().slice(0, 8)}`,
    documentId: docId,
    documentTitle: clonedDoc.facts.title,
    title: t.title,
    description: t.description,
    assigneeRole: t.assigneeRole,
    dueDateText: t.dueDateText,
    dueDateIso: t.dueDateIso,
    priority: t.priority,
    priorityRationale: t.priorityRationale,
    sourceQuote: t.sourceQuote,
    isAiRecommended: t.isAiRecommended,
    status: 'pending',
    createdAt: now,
    duplicateOfTaskId: t.id,
    duplicateNote: `Matches existing task "${t.title}" (100% match from duplicate notice).`,
    duplicateScore: 100,
    duplicateMatchingFields: ['Title', 'Assignee Role', 'Due Date'],
    translations: t.translations ? structuredClone(t.translations) : undefined,
  }));

  params.state.documents.unshift(clonedDoc);
  params.state.tasks.unshift(...clonedTasks);
  writeState(params.state);
  return { document: clonedDoc, createdTasks: clonedTasks };
}

async function createApp() {
  const app = express();
  app.use(express.json({ limit: '4mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    const key = process.env.GEMINI_API_KEY?.trim();
    res.json({
      ok: true,
      geminiConfigured: Boolean(key && key !== 'MY_GEMINI_API_KEY'),
      serverTime: new Date().toISOString(),
    });
  });

  app.get('/api/state', (_req: Request, res: Response) => {
    const state = readState();
    res.json(state);
  });

  app.post(
    '/api/documents/upload',
    upload.single('file'),
    async (req: Request, res: Response) => {
      try {
        const file = req.file;
        if (!file) {
          res.status(400).json({ error: 'No file provided. Select a PDF, JPG, PNG, or TXT file.' });
          return;
        }

        const allowDuplicate =
          req.body?.allowDuplicate === true || req.body?.allowDuplicate === 'true';
        const stateBeforeAnalysis = readState();

        const ext = path.extname(file.originalname).toLowerCase();
        let fileType: CampusDocument['fileType'] = 'txt';
        let extractionMethod: CampusDocument['extractionMethod'] = 'plain-text';
        let analysis;

        if (ext === '.txt' || file.mimetype === 'text/plain') {
          fileType = 'txt';
          extractionMethod = 'plain-text';
          const decodedText = file.buffer.toString('utf8').trim();
          if (!decodedText) {
            res.status(400).json({ error: 'The uploaded text file is empty.' });
            return;
          }
          const earlyHash = computeContentHash(decodedText);
          const exactExisting = stateBeforeAnalysis.documents.find(
            (d) => d.contentHash === earlyHash
          );
          if (exactExisting) {
            if (!allowDuplicate && stateBeforeAnalysis.settings.autoDuplicateCheck) {
              res.status(409).json({
                error: `Exact duplicate notice detected: matches "${exactExisting.facts.title}" (${exactExisting.fileName}).`,
                duplicateBlocked: true,
                existingDocument: exactExisting,
                similarityScore: 100,
                matchingFields: ['Exact Content Hash', 'Notice Body'],
                candidatePreview: {
                  fileName: file.originalname,
                  title: file.originalname.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' '),
                  issuedBy: exactExisting.facts.issuedBy,
                  issueDate: exactExisting.facts.issueDate,
                  summary: exactExisting.facts.summary,
                  extractedTaskCount: stateBeforeAnalysis.tasks.filter(
                    (t) => t.documentId === exactExisting.id
                  ).length,
                  rawTextExcerpt: decodedText.slice(0, 420),
                },
              });
              return;
            }
            if (allowDuplicate) {
              const cloned = cloneExactDuplicateForReview({
                state: stateBeforeAnalysis,
                existingDoc: exactExisting,
                fileName: file.originalname,
                fileType: 'txt',
                mimeType: file.mimetype || 'text/plain',
                fileSize: file.size,
                extractionMethod: 'plain-text',
              });
              res.status(201).json({
                document: cloned.document,
                createdTasks: cloned.createdTasks,
                state: stateBeforeAnalysis,
              });
              return;
            }
          }
          analysis = await analyzeDocumentWithGemini({
            textInput: decodedText,
            fileName: file.originalname,
          });
        } else if (ext === '.pdf' || file.mimetype === 'application/pdf') {
          fileType = 'pdf';
          let pdfText = '';
          try {
            pdfText = await extractTextFromPdfBuffer(file.buffer);
          } catch {
            pdfText = '';
          }

          if (pdfText.replace(/\s+/g, '').length >= 40) {
            extractionMethod = 'pdf-parse';
            const earlyHash = computeContentHash(pdfText);
            const exactExisting = stateBeforeAnalysis.documents.find(
              (d) => d.contentHash === earlyHash
            );
            if (exactExisting) {
              if (!allowDuplicate && stateBeforeAnalysis.settings.autoDuplicateCheck) {
                res.status(409).json({
                  error: `Exact duplicate PDF detected: matches "${exactExisting.facts.title}" (${exactExisting.fileName}).`,
                  duplicateBlocked: true,
                  existingDocument: exactExisting,
                  similarityScore: 100,
                  matchingFields: ['Exact Content Hash', 'PDF Text'],
                  candidatePreview: {
                    fileName: file.originalname,
                    title: file.originalname.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' '),
                    issuedBy: exactExisting.facts.issuedBy,
                    issueDate: exactExisting.facts.issueDate,
                    summary: exactExisting.facts.summary,
                    extractedTaskCount: stateBeforeAnalysis.tasks.filter(
                      (t) => t.documentId === exactExisting.id
                    ).length,
                    rawTextExcerpt: pdfText.slice(0, 420),
                  },
                });
                return;
              }
              if (allowDuplicate) {
                const cloned = cloneExactDuplicateForReview({
                  state: stateBeforeAnalysis,
                  existingDoc: exactExisting,
                  fileName: file.originalname,
                  fileType: 'pdf',
                  mimeType: file.mimetype || 'application/pdf',
                  fileSize: file.size,
                  extractionMethod: 'pdf-parse',
                });
                res.status(201).json({
                  document: cloned.document,
                  createdTasks: cloned.createdTasks,
                  state: stateBeforeAnalysis,
                });
                return;
              }
            }
            analysis = await analyzeDocumentWithGemini({
              textInput: pdfText,
              fileName: file.originalname,
            });
          } else {
            // Scanned/image-only PDF fallback via Gemini multimodal
            extractionMethod = 'gemini-multimodal-pdf';
            analysis = await analyzeDocumentWithGemini({
              binaryBuffer: file.buffer,
              mimeType: 'application/pdf',
              fileName: file.originalname,
            });
          }
        } else if (
          ext === '.jpg' ||
          ext === '.jpeg' ||
          ext === '.png' ||
          file.mimetype === 'image/jpeg' ||
          file.mimetype === 'image/png'
        ) {
          fileType = 'image';
          extractionMethod = 'gemini-multimodal-image';
          const mimeType =
            file.mimetype === 'image/png' || ext === '.png' ? 'image/png' : 'image/jpeg';
          analysis = await analyzeDocumentWithGemini({
            binaryBuffer: file.buffer,
            mimeType,
            fileName: file.originalname,
          });
        } else {
          res.status(400).json({ error: 'Unsupported file format.' });
          return;
        }

        const state = readState();
        const now = new Date().toISOString();
        const docId = `doc-${crypto.randomUUID().slice(0, 8)}`;
        const contentHash = computeContentHash(analysis.rawText || file.originalname);

        // Compare extracted content against existing notices to prevent accidental duplicates
        if (!allowDuplicate && state.settings.autoDuplicateCheck && state.documents.length > 0) {
          const docDup = findPotentialDocumentDuplicate(
            {
              contentHash,
              rawText: analysis.rawText,
              title: analysis.facts.title,
              issuedBy: analysis.facts.issuedBy,
              issueDate: analysis.facts.issueDate,
            },
            state.documents
          );
          if (docDup) {
            res.status(409).json({
              error: docDup.reason,
              duplicateBlocked: true,
              existingDocument: docDup.existingDocument,
              similarityScore: docDup.score,
              matchingFields: docDup.matchingFields,
              candidatePreview: {
                fileName: file.originalname,
                title: analysis.facts.title,
                issuedBy: analysis.facts.issuedBy,
                issueDate: analysis.facts.issueDate,
                summary: analysis.facts.summary,
                extractedTaskCount: analysis.extractedTasks.length,
                rawTextExcerpt: analysis.rawText.slice(0, 420),
              },
            });
            return;
          }
        }

        const newDoc: CampusDocument = {
          id: docId,
          fileName: file.originalname,
          fileType,
          mimeType: file.mimetype || 'application/octet-stream',
          fileSize: file.size,
          uploadedAt: now,
          extractionMethod,
          rawText: analysis.rawText,
          contentHash,
          facts: analysis.facts,
          aiRecommendations: analysis.aiRecommendations,
          translations: {
            en: {
              language: 'en',
              title: analysis.facts.title,
              summary: analysis.facts.summary,
              description: analysis.facts.description,
              targetAudience: analysis.facts.targetAudience,
              keyActionItems: analysis.extractedTasks.map(
                (t) =>
                  `${t.title}${
                    t.dueDateText && t.dueDateText !== 'Unknown'
                      ? ` (Due: ${t.dueDateText})`
                      : ' (Due: Unknown)'
                  }`
              ),
              translatedAt: now,
            },
          },
          duplicateOfDocumentId: null,
          duplicateScore: null,
          duplicateReason: null,
        };

        const newTasks: CampusTask[] = analysis.extractedTasks.map((t) => ({
          id: `task-${crypto.randomUUID().slice(0, 8)}`,
          documentId: docId,
          documentTitle: analysis.facts.title,
          title: t.title,
          description: t.description,
          assigneeRole: t.assigneeRole,
          dueDateText: t.dueDateText,
          dueDateIso: t.dueDateIso,
          priority: t.priority,
          priorityRationale: t.priorityRationale,
          sourceQuote: t.sourceQuote,
          isAiRecommended: t.isAiRecommended,
          status: 'pending',
          createdAt: now,
          duplicateOfTaskId: null,
          duplicateNote: null,
          translations: {
            en: { title: t.title, description: t.description },
          },
        }));

        state.documents.unshift(newDoc);
        state.tasks.unshift(...newTasks);

        // Run duplicate detection if enabled or if user imported an override duplicate for merge review
        if ((state.settings.autoDuplicateCheck || allowDuplicate) && state.documents.length > 1) {
          try {
            const scan = await detectDuplicatesWithGemini(state.documents, state.tasks);
            applyScanResultsToState(state, scan);
          } catch {
            const docDup = findPotentialDocumentDuplicate(
              {
                contentHash: newDoc.contentHash,
                rawText: newDoc.rawText,
                title: newDoc.facts.title,
                issuedBy: newDoc.facts.issuedBy,
                issueDate: newDoc.facts.issueDate,
              },
              state.documents.filter((d) => d.id !== newDoc.id)
            );
            if (docDup) {
              newDoc.duplicateOfDocumentId = docDup.existingDocument.id;
              newDoc.duplicateScore = docDup.score;
              newDoc.duplicateReason = docDup.reason;
              newDoc.duplicateMatchingFields = docDup.matchingFields;
            }
          }
        }

        writeState(state);
        const savedDoc = state.documents.find((d) => d.id === docId) ?? newDoc;
        const savedTasks = state.tasks.filter((t) => t.documentId === docId);

        res.status(201).json({
          document: savedDoc,
          createdTasks: savedTasks,
          state,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to process uploaded document.';
        res.status(500).json({ error: message });
      }
    }
  );

  app.post('/api/documents/text', async (req: Request, res: Response) => {
    try {
      const rawText = typeof req.body?.rawText === 'string' ? req.body.rawText.trim() : '';
      const titleInput =
        typeof req.body?.title === 'string' && req.body.title.trim()
          ? req.body.title.trim()
          : 'Pasted_Campus_Notice.txt';
      const allowDuplicate =
        req.body?.allowDuplicate === true || req.body?.allowDuplicate === 'true';

      if (rawText.length < 15) {
        res.status(400).json({ error: 'Please provide at least 15 characters of notice text.' });
        return;
      }

      const fileName = titleInput.endsWith('.txt') ? titleInput : `${titleInput}.txt`;
      const stateBeforeAnalysis = readState();

      const earlyHash = computeContentHash(rawText);
      const exactExisting = stateBeforeAnalysis.documents.find(
        (d) => d.contentHash === earlyHash
      );
      if (exactExisting) {
        if (!allowDuplicate && stateBeforeAnalysis.settings.autoDuplicateCheck) {
          res.status(409).json({
            error: `Exact duplicate notice detected: matches "${exactExisting.facts.title}" (${exactExisting.fileName}).`,
            duplicateBlocked: true,
            existingDocument: exactExisting,
            similarityScore: 100,
            matchingFields: ['Exact Content Hash', 'Notice Body'],
            candidatePreview: {
              fileName,
              title: titleInput.replace(/\.txt$/i, '').replace(/[_-]+/g, ' '),
              issuedBy: exactExisting.facts.issuedBy,
              issueDate: exactExisting.facts.issueDate,
              summary: exactExisting.facts.summary,
              extractedTaskCount: stateBeforeAnalysis.tasks.filter(
                (t) => t.documentId === exactExisting.id
              ).length,
              rawTextExcerpt: rawText.slice(0, 420),
            },
          });
          return;
        }
        if (allowDuplicate) {
          const cloned = cloneExactDuplicateForReview({
            state: stateBeforeAnalysis,
            existingDoc: exactExisting,
            fileName,
            fileType: 'text',
            mimeType: 'text/plain',
            fileSize: Buffer.byteLength(rawText, 'utf8'),
            extractionMethod: 'plain-text',
          });
          res.status(201).json({
            document: cloned.document,
            createdTasks: cloned.createdTasks,
            state: stateBeforeAnalysis,
          });
          return;
        }
      }

      const analysis = await analyzeDocumentWithGemini({
        textInput: rawText,
        fileName,
      });

      const state = readState();
      const now = new Date().toISOString();
      const docId = `doc-${crypto.randomUUID().slice(0, 8)}`;
      const contentHash = computeContentHash(analysis.rawText);

      if (!allowDuplicate && state.settings.autoDuplicateCheck && state.documents.length > 0) {
        const docDup = findPotentialDocumentDuplicate(
          {
            contentHash,
            rawText: analysis.rawText,
            title: analysis.facts.title,
            issuedBy: analysis.facts.issuedBy,
            issueDate: analysis.facts.issueDate,
          },
          state.documents
        );
        if (docDup) {
          res.status(409).json({
            error: docDup.reason,
            duplicateBlocked: true,
            existingDocument: docDup.existingDocument,
            similarityScore: docDup.score,
            matchingFields: docDup.matchingFields,
            candidatePreview: {
              fileName,
              title: analysis.facts.title,
              issuedBy: analysis.facts.issuedBy,
              issueDate: analysis.facts.issueDate,
              summary: analysis.facts.summary,
              extractedTaskCount: analysis.extractedTasks.length,
              rawTextExcerpt: analysis.rawText.slice(0, 420),
            },
          });
          return;
        }
      }

      const newDoc: CampusDocument = {
        id: docId,
        fileName,
        fileType: 'text',
        mimeType: 'text/plain',
        fileSize: Buffer.byteLength(rawText, 'utf8'),
        uploadedAt: now,
        extractionMethod: 'plain-text',
        rawText: analysis.rawText,
        contentHash,
        facts: analysis.facts,
        aiRecommendations: analysis.aiRecommendations,
        translations: {
          en: {
            language: 'en',
            title: analysis.facts.title,
            summary: analysis.facts.summary,
            description: analysis.facts.description,
            targetAudience: analysis.facts.targetAudience,
            keyActionItems: analysis.extractedTasks.map(
              (t) =>
                `${t.title}${
                  t.dueDateText && t.dueDateText !== 'Unknown'
                    ? ` (Due: ${t.dueDateText})`
                    : ' (Due: Unknown)'
                }`
            ),
            translatedAt: now,
          },
        },
        duplicateOfDocumentId: null,
        duplicateScore: null,
        duplicateReason: null,
      };

      const newTasks: CampusTask[] = analysis.extractedTasks.map((t) => ({
        id: `task-${crypto.randomUUID().slice(0, 8)}`,
        documentId: docId,
        documentTitle: analysis.facts.title,
        title: t.title,
        description: t.description,
        assigneeRole: t.assigneeRole,
        dueDateText: t.dueDateText,
        dueDateIso: t.dueDateIso,
        priority: t.priority,
        priorityRationale: t.priorityRationale,
        sourceQuote: t.sourceQuote,
        isAiRecommended: t.isAiRecommended,
        status: 'pending',
        createdAt: now,
        duplicateOfTaskId: null,
        duplicateNote: null,
        translations: {
          en: { title: t.title, description: t.description },
        },
      }));

      state.documents.unshift(newDoc);
      state.tasks.unshift(...newTasks);

      if ((state.settings.autoDuplicateCheck || allowDuplicate) && state.documents.length > 1) {
        try {
          const scan = await detectDuplicatesWithGemini(state.documents, state.tasks);
          applyScanResultsToState(state, scan);
        } catch {
          const docDup = findPotentialDocumentDuplicate(
            {
              contentHash: newDoc.contentHash,
              rawText: newDoc.rawText,
              title: newDoc.facts.title,
              issuedBy: newDoc.facts.issuedBy,
              issueDate: newDoc.facts.issueDate,
            },
            state.documents.filter((d) => d.id !== newDoc.id)
          );
          if (docDup) {
            newDoc.duplicateOfDocumentId = docDup.existingDocument.id;
            newDoc.duplicateScore = docDup.score;
            newDoc.duplicateReason = docDup.reason;
            newDoc.duplicateMatchingFields = docDup.matchingFields;
          }
        }
      }

      writeState(state);
      const savedDoc = state.documents.find((d) => d.id === docId) ?? newDoc;
      const savedTasks = state.tasks.filter((t) => t.documentId === docId);

      res.status(201).json({
        document: savedDoc,
        createdTasks: savedTasks,
        state,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to analyze pasted notice.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/documents/:id/translate', async (req: Request, res: Response) => {
    try {
      const docId = String(req.params.id);
      const lang = req.body?.language as SummaryLanguage;
      const force = Boolean(req.body?.force);

      if (!['en', 'kn', 'hi'].includes(lang)) {
        res.status(400).json({ error: 'Invalid target language. Supported: en, kn, hi.' });
        return;
      }

      const state = readState();
      const doc = state.documents.find((d) => d.id === docId);
      if (!doc) {
        res.status(404).json({ error: 'Document not found.' });
        return;
      }

      const docTasks = state.tasks.filter((t) => t.documentId === doc.id);
      const missingTaskTranslation = docTasks.some((t) => !t.translations?.[lang]);

      if (!force && doc.translations[lang] && !missingTaskTranslation) {
        res.json({ translation: doc.translations[lang], document: doc, state });
        return;
      }

      const translation = await translateDocumentWithGemini(doc, docTasks, lang);
      doc.translations[lang] = translation;

      if (translation.translatedTasks && translation.translatedTasks.length > 0) {
        const transMap = new Map(translation.translatedTasks.map((tt) => [tt.taskId, tt]));
        for (const task of state.tasks) {
          const match = transMap.get(task.id);
          if (match) {
            task.translations = {
              ...(task.translations ?? {}),
              [lang]: { title: match.title, description: match.description },
            };
          }
        }
      }

      writeState(state);

      res.json({ translation, document: doc, state });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Translation failed.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/documents/:id/reanalyze', async (req: Request, res: Response) => {
    try {
      const docId = String(req.params.id);
      const state = readState();
      const doc = state.documents.find((d) => d.id === docId);
      if (!doc) {
        res.status(404).json({ error: 'Document not found.' });
        return;
      }

      const analysis = await analyzeDocumentWithGemini({
        textInput: doc.rawText,
        fileName: doc.fileName,
      });

      const now = new Date().toISOString();
      doc.facts = analysis.facts;
      doc.aiRecommendations = analysis.aiRecommendations;
      doc.translations = {
        en: {
          language: 'en',
          title: analysis.facts.title,
          summary: analysis.facts.summary,
          description: analysis.facts.description,
          targetAudience: analysis.facts.targetAudience,
          keyActionItems: analysis.extractedTasks.map(
            (t) =>
              `${t.title}${
                t.dueDateText && t.dueDateText !== 'Unknown'
                  ? ` (Due: ${t.dueDateText})`
                  : ' (Due: Unknown)'
              }`
          ),
          translatedAt: now,
        },
      };

      // Replace old tasks from this doc with freshly extracted ones
      state.tasks = state.tasks.filter((t) => t.documentId !== doc.id);
      const refreshedTasks: CampusTask[] = analysis.extractedTasks.map((t) => ({
        id: `task-${crypto.randomUUID().slice(0, 8)}`,
        documentId: doc.id,
        documentTitle: analysis.facts.title,
        title: t.title,
        description: t.description,
        assigneeRole: t.assigneeRole,
        dueDateText: t.dueDateText,
        dueDateIso: t.dueDateIso,
        priority: t.priority,
        priorityRationale: t.priorityRationale,
        sourceQuote: t.sourceQuote,
        isAiRecommended: t.isAiRecommended,
        status: 'pending',
        createdAt: now,
        duplicateOfTaskId: null,
        duplicateNote: null,
      }));
      state.tasks.unshift(...refreshedTasks);

      writeState(state);
      res.json({ document: doc, createdTasks: refreshedTasks, state });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Re-analysis failed.';
      res.status(500).json({ error: message });
    }
  });

  app.delete('/api/documents/:id', (req: Request, res: Response) => {
    const docId = String(req.params.id);
    const state = readState();
    const exists = state.documents.some((d) => d.id === docId);
    if (!exists) {
      res.status(404).json({ error: 'Document not found.' });
      return;
    }

    const removedTaskIds = new Set(
      state.tasks.filter((t) => t.documentId === docId).map((t) => t.id)
    );
    state.documents = state.documents.filter((d) => d.id !== docId);
    state.tasks = state.tasks.filter((t) => t.documentId !== docId);
    // Clear dangling duplicate references on documents & tasks
    for (const d of state.documents) {
      if (d.duplicateOfDocumentId === docId) {
        d.duplicateOfDocumentId = null;
        d.duplicateScore = null;
        d.duplicateReason = null;
        d.duplicateMatchingFields = undefined;
      }
    }
    for (const t of state.tasks) {
      if (t.duplicateOfTaskId && removedTaskIds.has(t.duplicateOfTaskId)) {
        t.duplicateOfTaskId = null;
        t.duplicateNote = null;
        t.duplicateScore = null;
        t.duplicateMatchingFields = undefined;
      }
    }
    writeState(state);
    res.json({ state });
  });

  app.post('/api/documents/:id/dismiss-duplicate', (req: Request, res: Response) => {
    const docId = String(req.params.id);
    const state = readState();
    const doc = state.documents.find((d) => d.id === docId);
    if (!doc) {
      res.status(404).json({ error: 'Document not found.' });
      return;
    }
    doc.duplicateOfDocumentId = null;
    doc.duplicateScore = null;
    doc.duplicateReason = null;
    doc.duplicateMatchingFields = undefined;
    writeState(state);
    res.json({ state });
  });

  app.post('/api/documents/:id/merge', (req: Request, res: Response) => {
    const duplicateDocId = String(req.params.id);
    const targetDocumentId =
      typeof req.body?.targetDocumentId === 'string' ? req.body.targetDocumentId : '';
    const strategy =
      req.body?.strategy === 'replace_with_source' ? 'replace_with_source' : 'keep_target';
    const mergeUniqueTasks = req.body?.mergeUniqueTasks !== false;

    const state = readState();
    const sourceDoc = state.documents.find((d) => d.id === duplicateDocId);
    const targetDoc = state.documents.find((d) => d.id === targetDocumentId);

    if (!sourceDoc || !targetDoc || sourceDoc.id === targetDoc.id) {
      res.status(400).json({ error: 'Valid source and target documents are required to merge.' });
      return;
    }

    if (strategy === 'replace_with_source') {
      targetDoc.fileName = sourceDoc.fileName;
      targetDoc.fileType = sourceDoc.fileType;
      targetDoc.mimeType = sourceDoc.mimeType;
      targetDoc.fileSize = sourceDoc.fileSize;
      targetDoc.extractionMethod = sourceDoc.extractionMethod;
      targetDoc.rawText = sourceDoc.rawText;
      targetDoc.contentHash = sourceDoc.contentHash;
      targetDoc.facts = sourceDoc.facts;
      targetDoc.aiRecommendations = sourceDoc.aiRecommendations;
      targetDoc.translations = {
        ...targetDoc.translations,
        ...sourceDoc.translations,
      };
    } else {
      // Preserve any translations present on sourceDoc that targetDoc lacked
      targetDoc.translations = {
        ...sourceDoc.translations,
        ...targetDoc.translations,
      };
    }

    // Clear duplicate flag on surviving document
    targetDoc.duplicateOfDocumentId = null;
    targetDoc.duplicateScore = null;
    targetDoc.duplicateReason = null;
    targetDoc.duplicateMatchingFields = undefined;

    const targetTasks = state.tasks.filter((t) => t.documentId === targetDoc.id);
    const sourceTasks = state.tasks.filter((t) => t.documentId === sourceDoc.id);
    const tasksToDelete = new Set<string>();
    let movedUniqueTasks = 0;

    for (const sTask of sourceTasks) {
      const explicitMatch = sTask.duplicateOfTaskId
        ? targetTasks.find((tt) => tt.id === sTask.duplicateOfTaskId)
        : undefined;
      const similarityMatch =
        explicitMatch ??
        findPotentialTaskDuplicate(
          {
            title: sTask.title,
            description: sTask.description,
            assigneeRole: sTask.assigneeRole,
            dueDateIso: sTask.dueDateIso,
            dueDateText: sTask.dueDateText,
          },
          targetTasks
        )?.existingTask;

      if (similarityMatch) {
        // Consolidate completion/priority improvements onto surviving target task, then drop duplicate
        if (STATUS_RANK[sTask.status] > STATUS_RANK[similarityMatch.status]) {
          similarityMatch.status = sTask.status;
        }
        if (PRIORITY_RANK[sTask.priority] > PRIORITY_RANK[similarityMatch.priority]) {
          similarityMatch.priority = sTask.priority;
        }
        if (similarityMatch.dueDateText === 'Unknown' && sTask.dueDateText !== 'Unknown') {
          similarityMatch.dueDateText = sTask.dueDateText;
          similarityMatch.dueDateIso = sTask.dueDateIso;
        }
        similarityMatch.duplicateOfTaskId = null;
        similarityMatch.duplicateNote = null;
        similarityMatch.duplicateScore = null;
        similarityMatch.duplicateMatchingFields = undefined;
        tasksToDelete.add(sTask.id);
      } else if (mergeUniqueTasks) {
        // Reassign unique task from duplicate notice onto the surviving notice
        sTask.documentId = targetDoc.id;
        sTask.documentTitle = targetDoc.facts.title;
        sTask.duplicateOfTaskId = null;
        sTask.duplicateNote = null;
        sTask.duplicateScore = null;
        sTask.duplicateMatchingFields = undefined;
        movedUniqueTasks += 1;
      } else {
        tasksToDelete.add(sTask.id);
      }
    }

    // Remove merged-away duplicate document and redundant tasks
    state.documents = state.documents.filter((d) => d.id !== sourceDoc.id);
    state.tasks = state.tasks.filter((t) => !tasksToDelete.has(t.id));

    // Re-point or clear any remaining dangling duplicate references
    for (const d of state.documents) {
      if (d.duplicateOfDocumentId === sourceDoc.id) {
        d.duplicateOfDocumentId = targetDoc.id;
      }
    }
    for (const t of state.tasks) {
      if (t.duplicateOfTaskId && tasksToDelete.has(t.duplicateOfTaskId)) {
        t.duplicateOfTaskId = null;
        t.duplicateNote = null;
        t.duplicateScore = null;
        t.duplicateMatchingFields = undefined;
      }
    }

    writeState(state);
    res.json({
      state,
      mergedDocument: targetDoc,
      removedDocumentId: sourceDoc.id,
      movedUniqueTasks,
      removedDuplicateTasks: tasksToDelete.size,
    });
  });

  app.post('/api/duplicates/scan', async (_req: Request, res: Response) => {
    try {
      const state = readState();
      clearAllDuplicateFlags(state);

      const scan = await detectDuplicatesWithGemini(state.documents, state.tasks);
      applyScanResultsToState(state, scan);

      writeState(state);
      res.json({
        state,
        documentDuplicatesFound: scan.documentDuplicates.length,
        taskDuplicatesFound: scan.taskDuplicates.length,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Duplicate scan failed.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/tasks/translate', async (req: Request, res: Response) => {
    try {
      const lang = req.body?.language as SummaryLanguage;
      if (!['en', 'kn', 'hi'].includes(lang)) {
        res.status(400).json({ error: 'Invalid target language. Supported: en, kn, hi.' });
        return;
      }

      const state = readState();
      const missingTasks = state.tasks.filter((t) => !t.translations?.[lang]);

      if (missingTasks.length === 0) {
        res.json({ state, translatedCount: 0 });
        return;
      }

      const translated = await translateTasksWithGemini(missingTasks, lang);
      const transMap = new Map(translated.map((item) => [item.taskId, item]));
      for (const task of state.tasks) {
        const match = transMap.get(task.id);
        if (match) {
          task.translations = {
            ...(task.translations ?? {}),
            [lang]: { title: match.title, description: match.description },
          };
        }
      }

      writeState(state);
      res.json({ state, translatedCount: translated.length });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Task translation failed.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/tasks', (req: Request, res: Response) => {
    const title = typeof req.body?.title === 'string' ? req.body.title.trim() : '';
    if (!title) {
      res.status(400).json({ error: 'Task title is required.' });
      return;
    }

    const state = readState();
    const allowDuplicate = req.body?.allowDuplicate === true;
    const documentId =
      typeof req.body?.documentId === 'string' && req.body.documentId
        ? req.body.documentId
        : 'manual';
    const linkedDoc = state.documents.find((d) => d.id === documentId);

    const priority: PriorityLevel = ['high', 'medium', 'low'].includes(req.body?.priority)
      ? req.body.priority
      : 'medium';
    const assigneeRole: TargetRole = [
      'Student',
      'Teacher',
      'Administrator',
      'All',
      'Unknown',
    ].includes(req.body?.assigneeRole)
      ? req.body.assigneeRole
      : 'All';
    const dueDateIso =
      typeof req.body?.dueDateIso === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(req.body.dueDateIso)
        ? req.body.dueDateIso
        : null;
    const dueDateText =
      typeof req.body?.dueDateText === 'string' && req.body.dueDateText.trim()
        ? req.body.dueDateText.trim()
        : dueDateIso || 'Unknown';
    const description =
      typeof req.body?.description === 'string' ? req.body.description.trim() : '';

    const dupMatch = findPotentialTaskDuplicate(
      { title, description, assigneeRole, dueDateIso, dueDateText },
      state.tasks
    );

    if (!allowDuplicate && state.settings.autoDuplicateCheck && dupMatch) {
      res.status(409).json({
        error: dupMatch.reason,
        duplicateBlocked: true,
        existingTask: dupMatch.existingTask,
        similarityScore: dupMatch.score,
        matchingFields: dupMatch.matchingFields,
      });
      return;
    }

    const newTask: CampusTask = {
      id: `task-${crypto.randomUUID().slice(0, 8)}`,
      documentId,
      documentTitle: linkedDoc ? linkedDoc.facts.title : 'Manual Entry',
      title,
      description,
      assigneeRole,
      dueDateText,
      dueDateIso,
      priority,
      priorityRationale: 'Manually created task by workspace user.',
      sourceQuote: linkedDoc ? 'Manually linked to notice.' : 'Created manually in Tasks view.',
      isAiRecommended: false,
      status: 'pending',
      createdAt: new Date().toISOString(),
      duplicateOfTaskId: dupMatch ? dupMatch.existingTask.id : null,
      duplicateNote: dupMatch ? dupMatch.reason : null,
      duplicateScore: dupMatch ? dupMatch.score : null,
      duplicateMatchingFields: dupMatch ? dupMatch.matchingFields : undefined,
      translations: {
        en: { title, description },
      },
    };

    state.tasks.unshift(newTask);
    writeState(state);
    res.status(201).json({ task: newTask, state });
  });

  app.patch('/api/tasks/:id', (req: Request, res: Response) => {
    const taskId = String(req.params.id);
    const state = readState();
    const task = state.tasks.find((t) => t.id === taskId);
    if (!task) {
      res.status(404).json({ error: 'Task not found.' });
      return;
    }

    if (
      typeof req.body?.status === 'string' &&
      ['pending', 'in_progress', 'completed'].includes(req.body.status)
    ) {
      task.status = req.body.status as TaskStatus;
    }
    if (
      typeof req.body?.priority === 'string' &&
      ['high', 'medium', 'low'].includes(req.body.priority)
    ) {
      task.priority = req.body.priority as PriorityLevel;
    }
    if (typeof req.body?.title === 'string' && req.body.title.trim()) {
      task.title = req.body.title.trim();
    }
    if (typeof req.body?.description === 'string') {
      task.description = req.body.description.trim();
    }
    if (
      typeof req.body?.assigneeRole === 'string' &&
      ['Student', 'Teacher', 'Administrator', 'All', 'Unknown'].includes(req.body.assigneeRole)
    ) {
      task.assigneeRole = req.body.assigneeRole as TargetRole;
    }
    if (typeof req.body?.dueDateText === 'string') {
      task.dueDateText = req.body.dueDateText.trim() || 'Unknown';
    }
    if (req.body?.dueDateIso === null) {
      task.dueDateIso = null;
    } else if (
      typeof req.body?.dueDateIso === 'string' &&
      /^\d{4}-\d{2}-\d{2}$/.test(req.body.dueDateIso)
    ) {
      task.dueDateIso = req.body.dueDateIso;
    }

    writeState(state);
    res.json({ task, state });
  });

  app.post('/api/tasks/:id/dismiss-duplicate', (req: Request, res: Response) => {
    const taskId = String(req.params.id);
    const state = readState();
    const task = state.tasks.find((t) => t.id === taskId);
    if (!task) {
      res.status(404).json({ error: 'Task not found.' });
      return;
    }
    task.duplicateOfTaskId = null;
    task.duplicateNote = null;
    task.duplicateScore = null;
    task.duplicateMatchingFields = undefined;
    writeState(state);
    res.json({ state });
  });

  app.post('/api/tasks/:id/merge', (req: Request, res: Response) => {
    const duplicateTaskId = String(req.params.id);
    const targetTaskId = typeof req.body?.targetTaskId === 'string' ? req.body.targetTaskId : '';
    const keepTitleFrom = req.body?.keepTitleFrom === 'source' ? 'source' : 'target';
    const combineDescriptions = Boolean(req.body?.combineDescriptions);
    const priorityChoice = ['highest', 'target', 'source'].includes(req.body?.priorityChoice)
      ? (req.body.priorityChoice as 'highest' | 'target' | 'source')
      : 'highest';
    const statusChoice = ['most_progressed', 'target', 'source'].includes(req.body?.statusChoice)
      ? (req.body.statusChoice as 'most_progressed' | 'target' | 'source')
      : 'most_progressed';

    const state = readState();
    const sourceTask = state.tasks.find((t) => t.id === duplicateTaskId);
    const targetTask = state.tasks.find((t) => t.id === targetTaskId);

    if (!sourceTask || !targetTask || sourceTask.id === targetTask.id) {
      res.status(400).json({ error: 'Valid source and target tasks are required to merge.' });
      return;
    }

    if (keepTitleFrom === 'source') {
      targetTask.title = sourceTask.title;
    }

    if (combineDescriptions) {
      const targetDesc = targetTask.description.trim();
      const sourceDesc = sourceTask.description.trim();
      if (sourceDesc && !targetDesc) {
        targetTask.description = sourceDesc;
      } else if (
        sourceDesc &&
        targetDesc &&
        sourceDesc.toLowerCase() !== targetDesc.toLowerCase()
      ) {
        targetTask.description = `${targetDesc} | ${sourceDesc}`;
      }
    }

    if (priorityChoice === 'source') {
      targetTask.priority = sourceTask.priority;
    } else if (priorityChoice === 'highest') {
      if (PRIORITY_RANK[sourceTask.priority] > PRIORITY_RANK[targetTask.priority]) {
        targetTask.priority = sourceTask.priority;
      }
    }

    if (statusChoice === 'source') {
      targetTask.status = sourceTask.status;
    } else if (statusChoice === 'most_progressed') {
      if (STATUS_RANK[sourceTask.status] > STATUS_RANK[targetTask.status]) {
        targetTask.status = sourceTask.status;
      }
    }

    if (targetTask.dueDateText === 'Unknown' && sourceTask.dueDateText !== 'Unknown') {
      targetTask.dueDateText = sourceTask.dueDateText;
      targetTask.dueDateIso = sourceTask.dueDateIso;
    }

    targetTask.translations = {
      ...sourceTask.translations,
      ...targetTask.translations,
      en: { title: targetTask.title, description: targetTask.description },
    };

    targetTask.duplicateOfTaskId = null;
    targetTask.duplicateNote = null;
    targetTask.duplicateScore = null;
    targetTask.duplicateMatchingFields = undefined;

    state.tasks = state.tasks.filter((t) => t.id !== sourceTask.id);
    for (const t of state.tasks) {
      if (t.duplicateOfTaskId === sourceTask.id) {
        t.duplicateOfTaskId = targetTask.id;
      }
    }

    writeState(state);
    res.json({
      state,
      mergedTask: targetTask,
      removedTaskId: sourceTask.id,
    });
  });

  app.delete('/api/tasks/:id', (req: Request, res: Response) => {
    const taskId = String(req.params.id);
    const state = readState();
    if (!state.tasks.some((t) => t.id === taskId)) {
      res.status(404).json({ error: 'Task not found.' });
      return;
    }
    state.tasks = state.tasks.filter((t) => t.id !== taskId);
    for (const t of state.tasks) {
      if (t.duplicateOfTaskId === taskId) {
        t.duplicateOfTaskId = null;
        t.duplicateNote = null;
        t.duplicateScore = null;
        t.duplicateMatchingFields = undefined;
      }
    }
    writeState(state);
    res.json({ state });
  });

  app.get('/api/tasks/export.csv', (req: Request, res: Response) => {
    const state = readState();
    let tasks = state.tasks;

    const status = typeof req.query.status === 'string' ? req.query.status : '';
    const priority = typeof req.query.priority === 'string' ? req.query.priority : '';
    const role = typeof req.query.role === 'string' ? req.query.role : '';
    const documentId = typeof req.query.documentId === 'string' ? req.query.documentId : '';

    if (status && status !== 'all') {
      tasks = tasks.filter((t) => t.status === status);
    }
    if (priority && priority !== 'all') {
      tasks = tasks.filter((t) => t.priority === priority);
    }
    if (role && role !== 'All') {
      tasks = tasks.filter((t) => t.assigneeRole === role || t.assigneeRole === 'All');
    }
    if (documentId && documentId !== 'all') {
      tasks = tasks.filter((t) => t.documentId === documentId);
    }

    const headers = [
      'Task ID',
      'Title',
      'Description',
      'Priority',
      'Priority Rationale',
      'Status',
      'Due Date (Text)',
      'Due Date (ISO)',
      'Assignee Role',
      'Source Document',
      'Source Evidence Quote',
      'Origin',
      'Duplicate Flag',
    ];

    const rows = tasks.map((t) =>
      [
        escapeCsvCell(t.id),
        escapeCsvCell(t.title),
        escapeCsvCell(t.description),
        escapeCsvCell(t.priority.toUpperCase()),
        escapeCsvCell(t.priorityRationale),
        escapeCsvCell(t.status),
        escapeCsvCell(t.dueDateText),
        escapeCsvCell(t.dueDateIso ?? 'Unknown'),
        escapeCsvCell(t.assigneeRole),
        escapeCsvCell(t.documentTitle),
        escapeCsvCell(t.sourceQuote),
        escapeCsvCell(t.isAiRecommended ? 'AI Recommendation' : 'Document Mandate'),
        escapeCsvCell(t.duplicateOfTaskId ? `Duplicate of ${t.duplicateOfTaskId}` : 'Unique'),
      ].join(',')
    );

    const csvContent = [headers.map(escapeCsvCell).join(','), ...rows].join('\r\n');
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="campusflow-tasks-${new Date().toISOString().slice(0, 10)}.csv"`
    );
    res.send(csvContent);
  });

  app.post('/api/chat', async (req: Request, res: Response) => {
    try {
      const question = typeof req.body?.question === 'string' ? req.body.question.trim() : '';
      const scopedDocumentId =
        typeof req.body?.scopedDocumentId === 'string' && req.body.scopedDocumentId
          ? req.body.scopedDocumentId
          : null;
      const rolePerspective =
        typeof req.body?.rolePerspective === 'string' && req.body.rolePerspective
          ? req.body.rolePerspective
          : 'All';

      if (!question) {
        res.status(400).json({ error: 'Please enter a question.' });
        return;
      }

      const state = readState();
      const userMsg: ChatMessage = {
        id: `msg-${crypto.randomUUID().slice(0, 8)}`,
        role: 'user',
        content: question,
        timestamp: new Date().toISOString(),
        scopedDocumentId,
      };

      const aiResult = await chatWithDocuments({
        question,
        documents: state.documents,
        tasks: state.tasks,
        history: state.chatHistory,
        scopedDocumentId,
        rolePerspective,
      });

      const assistantMsg: ChatMessage = {
        id: `msg-${crypto.randomUUID().slice(0, 8)}`,
        role: 'assistant',
        content: aiResult.answer,
        timestamp: new Date().toISOString(),
        scopedDocumentId,
        foundInDocuments: aiResult.foundInDocuments,
        citations: aiResult.citations,
        documentFactsSection: aiResult.documentFactsSection,
        aiRecommendationsSection: aiResult.aiRecommendationsSection,
      };

      const updatedState = appendChatMessages([userMsg, assistantMsg]);
      res.json({ message: assistantMsg, state: updatedState });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Chat query failed.';
      res.status(500).json({ error: message });
    }
  });

  app.delete('/api/chat', (_req: Request, res: Response) => {
    const state = readState();
    state.chatHistory = [];
    writeState(state);
    res.json({ state });
  });

  app.patch('/api/settings', (req: Request, res: Response) => {
    const state = readState();
    if (typeof req.body?.institutionName === 'string' && req.body.institutionName.trim()) {
      state.settings.institutionName = req.body.institutionName.trim();
    }
    if (
      typeof req.body?.defaultRoleFilter === 'string' &&
      ['All', 'Student', 'Teacher', 'Administrator'].includes(req.body.defaultRoleFilter)
    ) {
      state.settings.defaultRoleFilter = req.body.defaultRoleFilter;
    }
    if (
      typeof req.body?.defaultLanguage === 'string' &&
      ['en', 'kn', 'hi'].includes(req.body.defaultLanguage)
    ) {
      state.settings.defaultLanguage = req.body.defaultLanguage;
    }
    if (typeof req.body?.autoDuplicateCheck === 'boolean') {
      state.settings.autoDuplicateCheck = req.body.autoDuplicateCheck;
    }
    writeState(state);
    res.json({ settings: state.settings, state });
  });

  app.post('/api/reset-demo', (_req: Request, res: Response) => {
    const state = resetToSampleState();
    res.json({ state });
  });

  app.post('/api/clear-all', (_req: Request, res: Response) => {
    const state = clearAllState();
    res.json({ state });
  });

  // Multer & API error handler
  app.use((err: unknown, req: Request, res: Response, next: NextFunction) => {
    if (req.path.startsWith('/api/')) {
      const message = err instanceof Error ? err.message : 'Unexpected server error.';
      res.status(400).json({ error: message });
      return;
    }
    next(err);
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CampusFlow AI server running at http://localhost:${PORT}`);
  });
}

createApp().catch((err) => {
  console.error('Failed to start CampusFlow AI server:', err);
  process.exit(1);
});
