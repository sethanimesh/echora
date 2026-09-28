import type { PropsWithChildren } from "react";
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { colors } from "../theme";

type Props = PropsWithChildren<{
  open: boolean;
  title: string;
  onClose: () => void;
}>;

export function Sheet({ open, title, onClose, children }: Props) {
  const insets = useSafeAreaInsets();
  return (
    <Modal visible={open} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView
        style={styles.modal}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <Pressable accessibilityLabel="Close sheet" style={styles.scrim} onPress={onClose} />
        <View style={[styles.card, { paddingBottom: Math.max(18, insets.bottom) }]}>
          <View style={styles.handle} />
          <View style={styles.head}>
            <Text accessibilityRole="header" style={styles.title}>{title}</Text>
            <Pressable accessibilityRole="button" hitSlop={12} onPress={onClose} style={styles.close}>
              <Text style={styles.closeText}>Close</Text>
            </Pressable>
          </View>
          <ScrollView
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={styles.content}
            showsVerticalScrollIndicator={false}
          >
            {children}
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modal: { flex: 1, justifyContent: "flex-end" },
  scrim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(12, 28, 24, 0.42)" },
  card: {
    maxHeight: "88%",
    minHeight: "36%",
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    backgroundColor: colors.paper,
    paddingHorizontal: 20,
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: -6 },
  },
  handle: { alignSelf: "center", width: 42, height: 5, borderRadius: 3, backgroundColor: colors.line, marginTop: 9 },
  head: { minHeight: 62, flexDirection: "row", alignItems: "center", borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.line },
  title: { flex: 1, color: colors.ink, fontSize: 22, fontWeight: "700" },
  close: { minHeight: 44, justifyContent: "center", paddingLeft: 18 },
  closeText: { color: colors.green, fontSize: 16, fontWeight: "700" },
  content: { paddingVertical: 20, gap: 16 },
});
