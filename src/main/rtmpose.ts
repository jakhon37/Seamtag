// AI pose: RTMPose finds a person's keypoints inside each box it's given (lane B builds this).
import type { PoseAnswer, PoseRequest } from '../shared/skeleton'

export async function poseInBoxes(req: PoseRequest): Promise<PoseAnswer> {
  void req
  throw new Error("AI pose isn't ready yet")
}
