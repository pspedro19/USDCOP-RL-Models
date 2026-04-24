import React from "react";
import { Composition, Folder } from "remotion";
import { Pitch, pitchSchema } from "./compositions/Pitch";
import { PITCH_TOTAL_FRAMES, FPS } from "./data/timings-pitch";
import { Thesis, thesisSchema } from "./compositions/Thesis";
import { THESIS_TOTAL_FRAMES } from "./data/timings-thesis";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Folder name="Pitch">
        <Composition
          id="PitchFinal"
          component={Pitch}
          durationInFrames={PITCH_TOTAL_FRAMES}
          fps={FPS}
          width={1920}
          height={1080}
          schema={pitchSchema}
          defaultProps={{ variant: "final" as const, vertical: false }}
        />
        <Composition
          id="PitchRaw"
          component={Pitch}
          durationInFrames={PITCH_TOTAL_FRAMES}
          fps={FPS}
          width={1920}
          height={1080}
          schema={pitchSchema}
          defaultProps={{ variant: "raw" as const, vertical: false }}
        />
        <Composition
          id="PitchFinalVertical"
          component={Pitch}
          durationInFrames={PITCH_TOTAL_FRAMES}
          fps={FPS}
          width={1080}
          height={1920}
          schema={pitchSchema}
          defaultProps={{ variant: "final" as const, vertical: true }}
        />
      </Folder>

      <Folder name="Thesis">
        <Composition
          id="ThesisFinal"
          component={Thesis}
          durationInFrames={THESIS_TOTAL_FRAMES}
          fps={FPS}
          width={1920}
          height={1080}
          schema={thesisSchema}
          defaultProps={{ variant: "final" as const }}
        />
        <Composition
          id="ThesisRaw"
          component={Thesis}
          durationInFrames={THESIS_TOTAL_FRAMES}
          fps={FPS}
          width={1920}
          height={1080}
          schema={thesisSchema}
          defaultProps={{ variant: "raw" as const }}
        />
      </Folder>
    </>
  );
};
