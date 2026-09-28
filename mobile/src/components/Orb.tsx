import { useEffect, useMemo, useRef } from "react";
import { AccessibilityInfo, Animated, Easing, StyleSheet, View } from "react-native";

import { colors } from "../theme";

type Props = {
  state: "listening" | "talking";
  level?: number;
  size?: number;
};

/** A native counterpart to the web shader orb: decorative, local, and optional. */
export function Orb({ state, level = 0.3, size = 210 }: Props) {
  const breath = useRef(new Animated.Value(0)).current;
  const rotate = useRef(new Animated.Value(0)).current;
  const reducedRef = useRef(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then((enabled) => {
      reducedRef.current = enabled;
    });
    const subscription = AccessibilityInfo.addEventListener("reduceMotionChanged", (enabled) => {
      reducedRef.current = enabled;
      if (enabled) {
        breath.stopAnimation();
        rotate.stopAnimation();
      }
    });
    return () => subscription.remove();
  }, [breath, rotate]);

  useEffect(() => {
    if (reducedRef.current) return;
    breath.setValue(0);
    rotate.setValue(0);
    const breathing = Animated.loop(
      Animated.sequence([
        Animated.timing(breath, {
          toValue: 1,
          duration: state === "talking" ? 560 : 850,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
        Animated.timing(breath, {
          toValue: 0,
          duration: state === "talking" ? 520 : 900,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
      ]),
    );
    const turning = Animated.loop(
      Animated.timing(rotate, {
        toValue: 1,
        duration: state === "talking" ? 4300 : 6800,
        easing: Easing.linear,
        useNativeDriver: true,
      }),
    );
    breathing.start();
    turning.start();
    return () => {
      breathing.stop();
      turning.stop();
    };
  }, [breath, rotate, state]);

  const lobes = useMemo(() => [0, 1, 2, 3, 4, 5], []);
  const liveScale = 0.96 + Math.min(1, Math.max(0, level)) * 0.12;
  const animatedScale = breath.interpolate({ inputRange: [0, 1], outputRange: [liveScale, liveScale + 0.07] });
  const spin = rotate.interpolate({ inputRange: [0, 1], outputRange: ["0deg", "360deg"] });

  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={{ width: size, height: size }}>
      <Animated.View
        style={[
          styles.halo,
          { width: size, height: size, borderRadius: size / 2, transform: [{ scale: animatedScale }] },
        ]}
      />
      <Animated.View
        style={{
          width: size,
          height: size,
          transform: [{ rotate: spin }, { scale: animatedScale }],
        }}
      >
        {lobes.map((index) => (
          <View
            key={index}
            style={[
              styles.lobe,
              {
                width: size * 0.58,
                height: size * 0.58,
                borderRadius: size * 0.29,
                left: size * (index % 2 === 0 ? 0.08 : 0.34),
                top: size * (index < 2 ? 0.08 : index < 4 ? 0.26 : 0.36),
                backgroundColor: index % 3 === 0 ? colors.green : index % 3 === 1 ? colors.sage : colors.coral,
                opacity: 0.62,
              },
            ]}
          />
        ))}
        <View
          style={[
            styles.core,
            {
              width: size * 0.42,
              height: size * 0.42,
              borderRadius: size * 0.21,
              left: size * 0.29,
              top: size * 0.29,
            },
          ]}
        />
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  halo: {
    position: "absolute",
    backgroundColor: colors.sageLight,
    opacity: 0.46,
  },
  lobe: {
    position: "absolute",
  },
  core: {
    position: "absolute",
    backgroundColor: colors.paper,
    opacity: 0.55,
  },
});
