// Who owns Seamtag and on what terms, as the About window and the macOS About panel say it, and
// the files that hold the licences. They're at the top of the repository, THIRD-PARTY-NOTICES.txt
// once npm run build has made it (scripts/notices.mjs), and electron-builder.yml packs them into
// resources/legal, where the About window reads them once installed.

export const COPYRIGHT = 'Copyright © 2026 Jakhongir Nodirov'
export const TERMS = 'Free software under the GNU GPL version 3 only, with ABSOLUTELY NO WARRANTY.'
export const INCLUDES = 'Includes code from other projects under their own licences.'

/** What the About window's two buttons show: Seamtag's licence, and the notices of what it includes */
export const LEGAL_FILES = { licence: ['LICENSE'], notices: ['NOTICE.md', 'THIRD-PARTY-NOTICES.txt'] } as const
export type LegalKind = keyof typeof LEGAL_FILES
/** A file's text, or why it can't be shown */
export type LegalText = { name: string; text: string } | { name: string; missing: string }
