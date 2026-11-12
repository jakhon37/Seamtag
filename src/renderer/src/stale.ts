/**
 * Images whose label files changed since the Problems dock's check, each by the number of its
 * last change: a check takes in the changes made before it began, and one saved while it ran
 * stays out of date, for the check that follows.
 */
export class Stale {
  private marks = new Map<string, number>()
  private count = 0

  /** These images' label files changed */
  mark(names: string[]) {
    for (const name of names) this.marks.set(name, ++this.count)
  }
  has(name: string) {
    return this.marks.has(name)
  }
  get size() {
    return this.marks.size
  }
  /** A check begins: the images to read again, and how far the changes it takes in go */
  begin() {
    return { names: [...this.marks.keys()], upTo: this.count }
  }
  /** The check begun at `upTo` is in: the changes it took in are checked. True while others wait. */
  settle(upTo: number) {
    for (const [name, at] of this.marks) if (at <= upTo) this.marks.delete(name)
    return this.marks.size > 0
  }
}
