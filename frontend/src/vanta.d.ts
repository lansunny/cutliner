declare module 'vanta/dist/vanta.fog.min' {
  export default function FOG(options: Record<string, unknown>): {
    setOptions(options: Record<string, unknown>): void;
    destroy(): void;
    resize(): void;
    req: number;
    prevNow?: number;
    animationLoop(): void;
    renderer?: { dispose(): void };
  };
}
declare module 'three';
