import { memo, useEffect, useRef } from 'react';
import { Camera, Color, Mesh, PlaneGeometry, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderer } from 'three';
import FOG from 'vanta/dist/vanta.fog.min';
import { flowPalettes } from './flowPalettes';

// Real Vanta FOG rendering, not a video or a progress simulation. All assets are
// bundled locally. CSS provides a lively fallback when WebGL is unavailable.
const THREE = { Camera, Color, Mesh, PlaneGeometry, Scene, ShaderMaterial, Vector2, Vector3, WebGLRenderer };

export default memo(function FlowBackground({ paused, palette }: { paused: boolean; palette: number }) {
  const host = useRef<HTMLDivElement>(null);
  const effect = useRef<ReturnType<typeof FOG> | null>(null);
  const running = useRef(false);
  const paletteRef = useRef(palette);
  paletteRef.current = palette;

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    let instance: ReturnType<typeof FOG> | null = null;
    let lost = false;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const colors = flowPalettes[paletteRef.current].colors;
    const onLost = (event: Event) => { event.preventDefault(); lost = true; node.dataset.webgl = 'fallback'; if (effect.current) cancelAnimationFrame(effect.current.req); running.current = false; };
    try {
      instance = FOG({
        el: node, THREE, mouseControls: false, touchControls: false, gyroControls: false,
        highlightColor: colors[0], midtoneColor: colors[1], lowlightColor: colors[2], baseColor: colors[3],
        blurFactor: .53, speed: 1.35, zoom: .62,
        scale: Math.max(1.6, devicePixelRatio || 1), scaleMobile: Math.max(2.4, devicePixelRatio || 1),
        minHeight: 200, minWidth: 200,
      });
      effect.current = instance;
      running.current = true;
      node.dataset.webgl = node.querySelector('canvas') ? 'ready' : 'fallback';
      node.querySelector('canvas')?.addEventListener('webglcontextlost', onLost);
    } catch {
      node.dataset.webgl = 'fallback';
    }

    let elapsed = 0;
    const timer = window.setInterval(() => {
      if (!instance || lost || !running.current || reducedMotion.matches || document.hidden) return;
      elapsed += .18;
      const base = flowPalettes[paletteRef.current].colors;
      const cycle = Math.sin(elapsed / 17) * .045;
      const asColor = (value: number, offset: number) => new THREE.Color(value).offsetHSL(cycle + offset, 0, 0).getHex();
      instance.setOptions({ highlightColor: asColor(base[0], 0), midtoneColor: asColor(base[1], cycle * .25), lowlightColor: asColor(base[2], -cycle * .2), baseColor: asColor(base[3], 0) });
    }, 180);

    const dispose = () => {
      clearInterval(timer);
      node.querySelector('canvas')?.removeEventListener('webglcontextlost', onLost);
      // Vanta removes geometry/listeners but not its renderer; dispose both.
      const renderer = instance?.renderer;
      instance?.destroy(); renderer?.dispose();
      if (effect.current === instance) effect.current = null;
      running.current = false;
    };
    return dispose;
  }, []);

  useEffect(() => {
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => {
      const instance = effect.current;
      if (!instance) return;
      const shouldRun = !paused && !reducedMotion.matches && !document.hidden;
      if (shouldRun && !running.current) {
        instance.prevNow = undefined; instance.animationLoop(); running.current = true;
      } else if (!shouldRun && running.current) {
        cancelAnimationFrame(instance.req); running.current = false;
      }
    };
    sync(); document.addEventListener('visibilitychange', sync); reducedMotion.addEventListener('change', sync);
    return () => { document.removeEventListener('visibilitychange', sync); reducedMotion.removeEventListener('change', sync); };
  }, [paused]);

  useEffect(() => {
    const c = flowPalettes[palette].colors;
    const instance = effect.current;
    instance?.setOptions({ highlightColor: c[0], midtoneColor: c[1], lowlightColor: c[2], baseColor: c[3] });
    if (instance && !running.current) {
      instance.prevNow = undefined;
      instance.animationLoop();
      cancelAnimationFrame(instance.req);
    }
  }, [palette]);

  return <div className={`flow-world palette-${palette} ${paused ? 'is-paused' : ''}`} aria-hidden="true">
    <div className="flow-fallback"/>
    <div className="vanta-field" ref={host}/>
    <div className="flow-ribbon ribbon-one"/><div className="flow-ribbon ribbon-two"/>
    <div className="flow-vignette"/>
  </div>;
});
